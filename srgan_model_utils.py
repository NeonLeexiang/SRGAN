"""
    date:       2021/4/3 11:00 上午
    written by: neonleexiang
"""
from keras.applications.vgg19 import VGG19
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam


class VGG_Loss:
    def __init__(self, image_shape):
        self.image_shape = image_shape

    '''
        because we use the vgg19 as our generator network.
        so we do not need to create our own network but use the vgg19
        then we can only use the vgg19 to set our loss and train.
        
        but be careful we need to download the weight file to initiate the 
        vgg19 network.
    '''
    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))


def get_optimizer(learning_rate=1e-4):
    """

    :param learning_rate:
    :return:
    """
    '''
        class torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)[source]
        
        params (iterable) – 待优化参数的iterable或者是定义了参数组的dict
        lr (float, 可选) – 学习率（默认：1e-3）
        betas (Tuple[float, float], 可选) – 用于计算梯度以及梯度平方的运行平均值的系数（默认：0.9，0.999）
        eps (float, 可选) – 为了增加数值计算的稳定性而加到分母里的项（默认：1e-8）
        weight_decay (float, 可选) – 权重衰减（L2惩罚）（默认: 0）
        
        Adam 算法还有一个 step method
        step(closure) [source]
        进行单次优化 (参数更新).
    '''
    return Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
