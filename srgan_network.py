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

    x = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    '''
        keras.layers.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, center=True, 
            scale=True, beta_initializer='zeros', gamma_initializer='ones', moving_mean_initializer='zeros', 
            moving_variance_initializer='ones', beta_regularizer=None, gamma_regularizer=None, 
            beta_constraint=None, gamma_constraint=None)
            
        axis: 整数，需要标准化的轴 （通常是特征轴）。 例如，在 data_format="channels_first" 的 Conv2D 层之后， 在 BatchNormalization 中设置 axis=1。
        momentum: 移动均值和移动方差的动量。
        epsilon: 增加到方差的小的浮点数，以避免除以零。
        center: 如果为 True，把 beta 的偏移量加到标准化的张量上。 如果为 False， beta 被忽略。
        scale: 如果为 True，乘以 gamma。 如果为 False，gamma 不使用。 当下一层为线性层（或者例如 nn.relu）， 
               这可以被禁用，因为缩放将由下一层完成。
        beta_initializer: beta 权重的初始化方法。
        gamma_initializer: gamma 权重的初始化方法。
        moving_mean_initializer: 移动均值的初始化方法。
        moving_variance_initializer: 移动方差的初始化方法。
        beta_regularizer: 可选的 beta 权重的正则化方法。
        gamma_regularizer: 可选的 gamma 权重的正则化方法。
        beta_constraint: 可选的 beta 权重的约束方法。
        gamma_constraint: 可选的 gamma 权重的约束方法。
    '''
    x = BatchNormalization(momentum=0.5)(x)
    # PReLU means Parametric ReLu
    '''
        keras.layers.PReLU(alpha_initializer='zeros', alpha_regularizer=None, 
                           alpha_constraint=None, shared_axes=None)
        
        参数化的 ReLU。
        形式： f(x) = alpha * x for x < 0, f(x) = x for x >= 0, 其中 alpha 是一个可学习的数组，尺寸与 x 相同。
        
        alpha_initializer: 权重的初始化函数。
        alpha_regularizer: 权重的正则化方法。
        alpha_constraint: 权重的约束。
        shared_axes: 激活函数共享可学习参数的轴。 
            例如，如果输入特征图来自输出形状为 (batch, height, width, channels) 的 2D 卷积层，
            而且你希望跨空间共享参数，以便每个滤波器只有一组参数， 可设置 shared_axes=[1, 2]。
        
    '''
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
    '''
        keras.layers.UpSampling2D(size=(2, 2), data_format=None, interpolation='nearest')
            2D 输入的上采样层。
            沿着数据的行和列分别重复 size[0] 和 size[1] 次。
            
        size: 整数，或 2 个整数的元组。 行和列的上采样因子。
        data_format: 字符串， channels_last (默认) 或 channels_first 之一， 表示输入中维度的顺序。
                    channels_last 对应输入尺寸为  (batch, height, width, channels)， 
                    channels_first 对应输入尺寸为  (batch, channels, height, width)。 
                    它默认为从 Keras 配置文件 ~/.keras/keras.json 中 找到的 image_data_format 值。 
                    如果你从未设置它，将使用 "channels_last"。
        interpolation: 字符串，nearest 或 bilinear 之一。 注意 CNTK 暂不支持 bilinear upscaling， 以及对于 Theano，只可以使用 size=(2, 2)。
    '''
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
    '''
        keras.layers.LeakyReLU(alpha=0.3)
        
        带泄漏的 ReLU。
        当神经元未激活时，它仍允许赋予一个很小的梯度： f(x) = alpha * x for x < 0, f(x) = x for x >= 0.
    
        alpha: float >= 0。负斜率系数。
        
        ReLU 的缺点：
        训练的时候很”脆弱”，很容易就”die”了
        例如，一个非常大的梯度流过一个 ReLU 神经元，更新过参数之后，这个神经元再也不会对任何数据有激活现象了，那么这个神经元的梯度就永远都会是 0.
        如果 learning rate 很大，那么很有可能网络中的 40% 的神经元都”dead”了。
        
        ReLU是将所有的负值都设为零，相反，Leaky ReLU是给所有负值赋予一个非零斜率。Leaky ReLU激活函数是在声学模型（2013）中首次提出的。
        
        参数化修正线性单元（PReLU）
        PReLU可以看作是Leaky ReLU的一个变体。在PReLU中，负值部分的斜率是根据数据来定的，而非预先定义的。
        作者称，在ImageNet分类（2015，Russakovsky等）上，PReLU是超越人类分类水平的关键所在。
    '''
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
        '''
            tanh activation 能够解决 zero-centered 的输出问题。
        '''
        x = Activation('tanh')(x)

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
        '''
            see the declaration in the srgan torch network
            alpha means x / alpha when x < 0
        '''
        x = LeakyReLU(alpha=0.2)(x)

        x = Dense(1)(x)
        x = Activation('sigmoid')(x)

        discriminator_model = Model(inputs=input_x, outputs=x)
        return discriminator_model
