import math
import sklearn.metrics
import tensorflow as tf
import os
import numpy as np
from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow import keras
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from model import create_model_Fastetransformerblock, CB_loss, create_ablation, create_vit_classifier
import keras
import csv
import random
import matplotlib.pyplot as plt
import seaborn as sns

#  Microsoft BIG
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

train_path = './dataset/vit_bytes/train'
val_path = './dataset/vit_bytes/val'
img_width, img_height = 224, 224
img_size = (img_width, img_height)


def img_data_gen(imgs_path, img_size, batch_size, rescale, shuffle=False):
    return ImageDataGenerator(rescale=rescale).flow_from_directory(imgs_path,
                                                                   target_size=img_size,
                                                                   batch_size=batch_size,
                                                                   class_mode='categorical',
                                                                   shuffle=shuffle)

train_gen = img_data_gen(imgs_path=train_path,
                         img_size=img_size,
                         batch_size=7602,
                         rescale=1./255,
                         shuffle=True)

train_imgs, train_labels = next(train_gen)
print(f"imgs.shape:{train_imgs.shape},labels.shape:{train_labels.shape}")

val_gen = img_data_gen(imgs_path=val_path,
                         img_size=img_size,
                         batch_size=1629,
                         rescale=1./255,
                         shuffle=True)

val_imgs, val_labels = next(val_gen)
print(f"imgs.shape:{val_imgs.shape},labels.shape:{val_labels.shape}")

X_train = train_imgs
y_train = train_labels
X_val = val_imgs
y_val = val_labels

print(f"X_train.shape: {X_train.shape}, X_val.shape: {X_val.shape}")
print(f"y_train.shape: {y_train.shape}, y_val.shape: {y_val.shape}")


num_classes = train_labels.shape[1]
input_shape = train_imgs.shape[1:]
learning_rate = 0.0001
weight_decay = 0.00001
batch_size = 32
num_epochs = 30
patience = 10
image_size = 224  # We'll resize input images to this size
patch_size = 16
num_patches = (image_size // patch_size) ** 2
num_heads = 6
projection_dim = 64

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier


def cb_loss(labels, logits):
    samples_per_cls = [1073, 1735, 2059, 332, 29, 526, 279, 860, 709]
    classes = 9
    loss_type = "focal"
    beta = 0.8
    gamma = 2.5
    return CB_loss(labels, logits, samples_per_cls, classes, loss_type, beta, gamma)

def scheduler(now_epoch):
    end_lr_rate = 0.01
    rate = ((1 + math.cos(now_epoch * math.pi / num_epochs)) / 2) * (1 - end_lr_rate) + end_lr_rate
    new_lr = rate * learning_rate
    return new_lr


class SaveAccuracyCallback(keras.callbacks.Callback):
    def __init__(self, file_path, model_name):
        super(SaveAccuracyCallback, self).__init__()
        self.file_path = file_path
        self.model_name = model_name
        self.fieldnames = ['Model Name', 'Epoch', 'lr', 'Accuracy', 'Loss', 'Val Accuracy', 'Val Loss']

        if not os.path.exists(self.file_path):
            with open(self.file_path, 'w', newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
                writer.writeheader()

    def on_epoch_end(self, epoch, logs=None):
        lr = float(keras.backend.get_value(self.model.optimizer.lr))
        accuracy = logs.get('accuracy')
        loss = logs.get('loss')
        val_accuracy = logs.get('val_accuracy')
        val_loss = logs.get('val_loss')
        with open(self.file_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.fieldnames)
            writer.writerow({'Model Name': self.model_name,
                             'Epoch': epoch + 1,
                             'lr': lr,
                             'Accuracy': accuracy,
                             'Loss': loss,
                             'Val Accuracy': val_accuracy,
                             'Val Loss': val_loss
                             })


file_path = './logs/gaijin_pamaac.csv'
Model_name = "22fasterblock_ViT.h5"
# Model_name = "FS_nocbloss_ViT.h5"
save_accuracy_callback = SaveAccuracyCallback(file_path, model_name=Model_name)

def run_experiment(model):
    optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
    # optimizer = tf.optimizers.Adam(learning_rate=learning_rate)
    model.compile(
        optimizer=optimizer,
        # loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        # CB_loss
        loss=cb_loss,
        metrics=[
            keras.metrics.CategoricalAccuracy(name="accuracy"),
        ],
    )
    model_name = Model_name
    log_dir = os.path.join(os.getcwd(), 'logs')

    filepath = 'fasterblock.{epoch:02d}-{val_loss:.4f}.h5'
    ck_path = os.path.join(log_dir, filepath)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    mc = keras.callbacks.ModelCheckpoint(ck_path, monitor='val_loss',
                                         save_best_only=True,
                                         save_weights_only=True)
    es = keras.callbacks.EarlyStopping(monitor='val_loss',
                                       patience=patience,
                                       verbose=0)

    reduce_lr = LearningRateScheduler(scheduler)
    callbacks = [save_accuracy_callback, es, mc, reduce_lr]

    history = model.fit(
        x=X_train,
        y=y_train,
        batch_size=batch_size,
        epochs=num_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
    )

    print(history.history.keys())

    # save model
    model_path = os.path.join(log_dir, model_name)
    model.save(model_path)
    return history

vit_classifier = create_model_Fastertransformerblock(input_shape, patch_size, num_patches, projection_dim,
                                                    transformer_layers, transformer_units, mlp_head_units, num_classes)

# vit_classifier = create_vit_classifier(input_shape, patch_size, num_patches, projection_dim, num_heads, transformer_layers,
#                                        transformer_units, mlp_head_units, num_classes)


vit_classifier.summary()
# keras.utils.plot_model(vit_classifier, show_shapes=True)

history = run_experiment(vit_classifier)

test_path = './dataset/vit_bytes/test'
test_gen = img_data_gen(imgs_path=test_path,
                        img_size=img_size,
                        batch_size=1629,
                        rescale=1. / 255,
                        shuffle=True)

test_imgs, test_labels = next(test_gen)
X_test = test_imgs
y_test = test_labels
test_loss, test_accuracy = vit_classifier.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

y_pred = vit_classifier.predict(X_test)

y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

print(classification_report(y_true_classes, y_pred_classes, target_names=["Ramnit", "Lollipop", "Kelihos_ver3", "Vundo",
                                                                       "Simda", "Tracur", "Kelihos_ver1", "Obfuscator.ACY",
                                                                       "Gatak"], digits=5))

# 马修斯相关系数MCC
mcc = sklearn.metrics.matthews_corrcoef(y_true_classes, y_pred_classes)
print("马修斯系数：", mcc)
# 归一化混淆矩阵
conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)
normalized_conf_matrix = conf_matrix / class_totals
plt.figure(figsize=(10, 8))
sns.heatmap(normalized_conf_matrix, annot=True, fmt='.2f', cmap='Blues', cbar=True)
plt.xlabel('Predicted Class')
plt.ylabel('True Class')
# plt.title('Confusion Matrix')
plt.show()


def plot_history(history):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()
