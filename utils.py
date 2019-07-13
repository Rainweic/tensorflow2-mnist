# 该类提供工具函数以供使用

import csv
import cv2 as cv
import numpy as np
import tensorflow as tf

def get_train_images_from_csv(csv_path):
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
                # 第0个数值为label 后面的是图像像素值
                labels.append(row[0])
                images_list.append(row[1:])
    # 转为ndarray 设为float32可以之后用opencv显示
    labels = np.array(labels, dtype="float32")
    images_list = np.array(images_list, dtype="float32")
    # 将图片shape恢复到 [n, h, w, c] 的样子
    images_list = images_list.reshape((-1, 28, 28, 1))
    return images_list, labels

def preprocess(labels, images):
    """
    预处理函数
    """
    # 转为tensor 且对images进行归一化
    labels = tf.cast(labels, dtype=tf.int32)
    # 用fit函数不需要one_hot编码
    # labels = tf.one_hot(labels, depth=10)
    images = tf.cast(images, dtype=tf.float32) / 255
    return images, labels

def mnist_train_dataset():
    """
    加载数据集
    """
    image_list, labels = get_train_images_from_csv("./datasets/train.csv")
    # Step1 加载数据集
    dataset = tf.data.Dataset.from_tensor_slices((labels, image_list))
    # Step2 打乱数据
    dataset = dataset.shuffle(2000)
    # Step3 预处理图像
    dataset = dataset.map(preprocess)
    # Step4 设置batch size
    dataset = dataset.batch(64)
    return dataset

def show_image(label, image):
    cv.imshow(str(label), image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    