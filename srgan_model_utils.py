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

    # TODO: vgg19 and vgg loss
    def vgg_loss(self, y_true, y_pred):
        vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=self.image_shape)
        vgg19.trainable = False
        for layer in vgg19.layers:
            layer.trainable = False

        model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
        model.trainable = False

        return K.mean(K.square(model(y_true) - model(y_pred)))


def get_optimizer(learning_rate=1e-4):
    # TODO: adam parameters setting and declaration
    return Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-8)
