import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def small_resnet():
    '''
    小型残差网络
    '''
    inputs = keras.Input(shape=(28,28,1), name='img')
    h1 = layers.Conv2D(32, 3, activation='relu')(inputs)
    h1 = layers.Conv2D(64, 3, activation='relu')(h1)
    block1_out = layers.MaxPooling2D(3)(h1)

    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(block1_out)
    h2 = layers.Conv2D(64, 3, activation='relu', padding='same')(h2)
    block2_out = layers.add([h2, block1_out])

    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(block2_out)
    h3 = layers.Conv2D(64, 3, activation='relu', padding='same')(h3)
    block3_out = layers.add([h3, block2_out])

    h4 = layers.Conv2D(64, 3, activation='relu')(block3_out)
    h4 = layers.GlobalMaxPool2D()(h4)
    h4 = layers.Dense(256, activation='relu')(h4)
    h4 = layers.Dropout(0.5)(h4)
    outputs = layers.Dense(10, activation='softmax')(h4)

    model = keras.Model(inputs, outputs, name='small resnet')
    model.summary()
    keras.utils.plot_model(model, 'small_resnet_model.png', show_shapes=True)
    return model
    



