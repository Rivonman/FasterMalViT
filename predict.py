import csv
import os
import numpy as np
import sklearn
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import keras
import tensorflow_addons as tfa
from model import create_model_Fastetransformerblock, CB_loss
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
def img_data_gen(imgs_path, img_size, batch_size, rescale, shuffle=False):
    return ImageDataGenerator(rescale=rescale).flow_from_directory(imgs_path,
                                                                   target_size=img_size,
                                                                   batch_size=batch_size,
                                                                   class_mode='categorical',
                                                                   shuffle=shuffle)

# test file path
test_path = "./dataset/vit_bytes/test2"

learning_rate = 0.0001
weight_decay = 0.00001
batch_size = 32
num_epochs = 30
patience = 10
image_size = 224 # We'll resize input images to this size
patch_size = 16
num_patches = (image_size // patch_size) ** 2
num_heads = 6
projection_dim = 64

transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [2048, 1024]


test_gen = img_data_gen(imgs_path=test_path,
                        img_size=(224, 224),
                        batch_size=1629,
                        rescale=1. / 255,
                        shuffle=False)

test_imgs, test_labels = next(test_gen)
X_test = test_imgs
y_test = test_labels
num_classes = test_labels.shape[1]
input_shape = test_imgs.shape[1:]

# create model
vit_classifier = create_model_Fastetransformerblock(input_shape, patch_size, num_patches, projection_dim,
                                                    transformer_layers, transformer_units, mlp_head_units, num_classes)

# load model weight
vit_classifier.load_weights("./logs/19fasterblock.28-0.0763.h5")


def cb_loss(labels, logits):
    samples_per_cls = [1073, 1735, 2059, 332, 29, 526, 279, 860, 709]  # The number of samples per class in the training set.
    classes = 9
    loss_type = "focal"
    beta = 0.8
    gamma = 2.5 
    return CB_loss(labels, logits, samples_per_cls, classes, loss_type, beta, gamma)

optimizer = tfa.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
vit_classifier.compile(
    optimizer=optimizer,
    # CB_loss
    loss=cb_loss,
    metrics=[
    keras.metrics.CategoricalAccuracy(name="accuracy"),
    ],
)

# predict
predictions = vit_classifier.predict(X_test)
predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)
class_labels = list(test_gen.class_indices.keys())

print("Sample Predictions for True Class 5:")
for i in range(10):
    print(f"True Class: {class_labels[true_classes[i]]}, Predicted Class: {class_labels[predicted_classes[i]]}")


def print_and_write_all_class_probabilities(csv_filename="class_predictions.csv"):
    with open(csv_filename, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Sample Index', 'True Class', 'Predicted Class', 'Predicted Probabilities'])

        for i in range(len(true_classes)):
            true_class_label = class_labels[true_classes[i]]

            # Softmax
            probs = tf.nn.softmax(predictions[i])

            # write to CSV
            for j in range(len(probs)):
                writer.writerow([i, true_class_label, class_labels[j], probs[j].numpy()])

            print(f"\nSample {i} true class: '{true_class_label}' probabilities written to CSV.")


csv_filename = "class_predictions.csv"
print_and_write_all_class_probabilities(csv_filename)
