# 该类提供工具函数以供使用

import csv
import cv2 as cv
import numpy as np
import tensorflow as tf

def get_images_from_csv(csv_path, train=True):
    """
    从csv文件中读取numpy格式的image
    """
    # 存放图片与labels
    images_list, labels = [], []
    with open(csv_path, "r") as f:
        read = csv.reader(f)
        for i, row in enumerate(read):
            # 第一行非数值
            if i != 0:
                if train:
                    # 第0个数值为label 后面的是图像像素值
                    labels.append(row[0])
                    images_list.append(row[1:])
                else:
                    images_list.append(row)
    # 转为ndarray 设为float32可以之后用opencv显示
    labels = np.array(labels, dtype="float32")
    images_list = np.array(images_list, dtype="float32")
    # 将图片shape恢复到 [n, h, w, c] 的样子
    images_list = images_list.reshape((-1, 28, 28, 1))
    if train:
        return images_list, labels
    else:
        return images_list

def preprocess(labels=None, images=None, train=True):
    """
    预处理函数
    """
    if train:
        # 转为tensor 且对images进行归一化
        labels = tf.cast(labels, dtype=tf.int32)
    # 用fit函数不需要one_hot编码
    # labels = tf.one_hot(labels, depth=10)
    images = tf.cast(images, dtype=tf.float32) / 255
    # 减去均值
    image_mean = tf.reduce_mean(images, axis=0)
    images -= image_mean
    if train:
        return images, labels
    else:
        return images

def mnist_train_dataset():
    """
    加载数据集
    """
    image_list, labels = get_images_from_csv("./datasets/train.csv")
    # Step1 加载数据集
    dataset = tf.data.Dataset.from_tensor_slices((labels, image_list))
    # Step2 打乱数据
    dataset = dataset.shuffle(2000)
    # Step3 预处理图像
    dataset = dataset.map(preprocess)
    # Step4 设置batch size
    dataset = dataset.batch(64)
    return dataset

def write_to_csv(csv_path, datasets):
    with open(csv_path, 'a', newline="") as f:
        write = csv.writer(f,dialect='excel')
        write.writerow(["ImageId", "Label"])
        write.writerows(datasets)

def show_image(label, image):
    cv.imshow(str(label), image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    