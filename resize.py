import os
import cv2
import math
import numpy as np

def bi_linear(src, dst, target_size):
    pic = cv2.imread(src)
    th, tw = target_size[0], target_size[1]
    emptyImage = np.zeros(target_size, np.uint8)
    for k in range(3):
        for i in range(th):
            for j in range(tw):
                corr_x = (i + 0.5) / th * pic.shape[0] - 0.5
                corr_y = (j + 0.5) / tw * pic.shape[1] - 0.5

                point1 = (math.floor(corr_x), math.floor(corr_y))  # 左上角的点
                point2 = (point1[0], point1[1] + 1)
                point3 = (point1[0] + 1, point1[1])
                point4 = (point1[0] + 1, point1[1] + 1)

                fr1 = (point2[1] - corr_y) * pic[point1[0], point1[1], k] + (
                        corr_y - point1[1]) * pic[point2[0], point2[1], k]
                fr2 = (point2[1] - corr_y) * pic[point3[0], point3[1], k] + (
                        corr_y - point1[1]) * pic[point4[0], point4[1], k]
                emptyImage[i, j, k] = (point3[0] - corr_x) * fr1 + (corr_x - point1[0]) * fr2

    cv2.imwrite(dst, emptyImage)


def batch_resize(input_folder, output_folder, target_size):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for category_folder in os.listdir(input_folder):
        category_path = os.path.join(input_folder, category_folder)
        if os.path.isdir(category_path):
            output_category_folder = os.path.join(output_folder, category_folder)
            if not os.path.exists(output_category_folder):
                os.makedirs(output_category_folder)

            for filename in os.listdir(category_path):
                if filename.endswith(('.jpg', '.png', '.jpeg')): 
                    src = os.path.join(category_path, filename)
                    dst = os.path.join(output_category_folder, filename)
                    bi_linear(src, dst, target_size)

def main():
    input_folder = 'folder path'
    output_folder = 'target path'
    target_size = (224, 224, 3) 

    batch_resize(input_folder, output_folder, target_size)

if __name__ == '__main__':
    main()
