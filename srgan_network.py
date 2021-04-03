"""
    date:       2021/4/2 2:03 下午
    written by: neonleexiang
"""
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU, PReLU
from keras.layers import add, Input, Dense
from keras.layers.core import Activation, Flatten
from keras.models import Model


# Residual block
def res_block_generator(x, kernel_size, filters, strides):
    """

    :param x:
    :param kernel_size:
    :param filters:
    :param strides:
    :return:
    """
    input_x = x

    # TODO: 1. figure out momentum and BatchNormalization 2. PReLU
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    # PReLU means Parametric ReLu
    x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)

    x = add([input_x, x])
    return x


# up sampling block for generator
def up_sampling_block(x, kernel_size, filters, strides):
    """

    :param x:
    :param kernel_size:
    :param filters:
    :param strides:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    # TODO: what is UpSampling2D
    x = UpSampling2D(size=2)(x)
    x = LeakyReLU(alpha=0.2)(x)
    return x


def discriminator_block(x, filters, kernel_size, strides):
    """

    :param x:
    :param filters:
    :param kernel_size:
    :param strides:
    :return:
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization(momentum=0.5)(x)
    # TODO: why alpha=0.2 and what is alpha
    x = LeakyReLU(alpha=0.2)(x)
    return x


# building the generator
class Generator:
    def __init__(self, noise_shape):
        self.noise_shape = noise_shape

    def generator(self):
        input_x = Input(shape=self.noise_shape)
        x = Conv2D(filters=64, kernel_size=9, strides=1, padding='same')(input_x)
        x = PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=[1, 2])(x)
        res_x = x

        for _ in range(16):
            x = res_block_generator(x, 3, 64, 1)

        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(x)
        x = BatchNormalization(momentum=0.5)(x)
        x = add([res_x, x])

        for _ in range(2):
            x = up_sampling_block(x, 3, 256, 1)

        x = Conv2D(filters=3, kernel_size=9, strides=1, padding='same')(x)
        x = Activation('tanh')(x)       # TODO: why using tanh as activation

        generator_model = Model(inputs=input_x, outputs=x)
        return generator_model


class Discriminator:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    def discriminator(self):
        input_x = Input(shape=self.image_shape)

        x = Conv2D(filters=64, kernel_size=3, strides=1, padding='same')(input_x)
        x = LeakyReLU(alpha=0.2)(x)

        x = discriminator_block(x, 64, 3, 2)
        x = discriminator_block(x, 128, 3, 1)
        x = discriminator_block(x, 128, 3, 2)
        x = discriminator_block(x, 256, 3, 1)
        x = discriminator_block(x, 256, 3, 2)
        x = discriminator_block(x, 512, 3, 1)
        x = discriminator_block(x, 512, 3, 2)

        x = Flatten()(x)
        x = Dense(1024)(x)
        x = LeakyReLU(alpha=0.2)(x)     # TODO: the declaration of LeakyReLU

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        discriminator_model = Model(inputs=input_x, outputs=x)
        return discriminator_model
