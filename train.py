from src import utils
from model import resnet
import tensorflow as tf
from tensorflow import keras

def train():
    # 加载数据集
    train_dataset = utils.mnist_train_dataset()
    # 加载模型
    model = resnet.small_resnet()

    # 编译网络模型 设置optimizer loss metrics
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate = 0.0001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    callback = [
        # 使用tensorboard
        keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    model.fit(
        train_dataset,
        epochs = 30,
        callbacks = callback
    )

if __name__ == "__main__":
    train()
