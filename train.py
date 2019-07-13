import utils
from model import resnet
from tensorflow import keras

def train():
    # 加载数据集
    train_dataset = utils.mnist_train_dataset()
    # 加载模型
    model = resnet.small_resnet()

    # 编译网络模型 设置optimizer loss metrics
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate = 0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

    # 使用tensorboard
    tensorboard_cbk = keras.callbacks.TensorBoard(log_dir='./full_path_to_your_logs')
    # Checkpoint保存权重文件
    check_callback = keras.callbacks.ModelCheckpoint(
        filepath='./weights/mymodel_{epoch}.h5',
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    model.fit(
        train_dataset,
        epochs = 10,
        callbacks = [tensorboard_cbk, check_callback]
    )

if __name__ == "__main__":
    train()
