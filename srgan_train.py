"""
    date:       2021/4/3 3:37 下午
    written by: neonleexiang
"""
import numpy as np
from keras.layers import Input
from keras.models import Model
from srgan_network import Generator, Discriminator
import srgan_model_utils
import srgan_utils
from tqdm import tqdm
from datetime import datetime
import tensorflow as tf


np.random.seed(34)
downscale_factor = 4
image_shape = (256, 256, 3)


def mse(y, t):
    return np.mean(np.square(y - t))


def psnr(y, t):
    return 20 * np.log10(255) - 10 * np.log10(mse(y, t))


# Combined network
def gan_network(generator, discriminator, shape, optimizer, loss):
    discriminator.trainable = False
    gan_input = Input(shape=shape)
    generator_output = generator(gan_input)
    gan_output = discriminator(generator_output)
    gan_model = Model(inputs=gan_input, outputs=[generator_output, gan_output])
    gan_model.compile(loss=[loss, 'binary_crossentropy'], loss_weights=[1., 1e-3], optimizer=optimizer)
    return gan_model


def train(epochs=5, batch_size=2, learning_rate=1e-4, img_shape=image_shape,
          train_dir='datasets/train/', val_dir='datasets/val/', model_save_dir='srgan_model_file/'):
    print('-'*15+'> start loading data... ', '[ {} ]'.format(datetime.now()))
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = srgan_utils.load_train_val_data(train_dir, val_dir)

    # x_train_lr = x_train_lr[:4]
    # x_train_hr = x_train_hr[:4]
    # x_test_lr = x_test_lr[:2]
    # x_test_hr = x_test_hr[:2]

    print('-'*15 + '> data loading......')
    print(x_train_lr.shape)
    print(x_train_hr.shape)
    print(x_test_lr.shape)
    print(x_test_hr.shape)

    # loss = srgan_model_utils.VGG_Loss(img_shape)
    loss = 'mse'

    batch_count = int(x_train_hr.shape[0] / batch_size)
    lr_shape = (x_train_lr.shape[1], x_train_lr.shape[2], x_train_lr.shape[3])

    generator = Generator(lr_shape).generator()
    discriminator = Discriminator(img_shape).discriminator()

    optimizer = srgan_model_utils.get_optimizer(learning_rate)
    # generator.compile(loss=loss.vgg_loss, optimizer=optimizer)
    generator.compile(loss=loss, optimizer=optimizer)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    # loss.vgg_loss is unused
    # gan = gan_network(generator=generator, discriminator=discriminator,
    #                   shape=lr_shape, optimizer=optimizer, loss=loss.vgg_loss)
    gan = gan_network(generator=generator, discriminator=discriminator,
                      shape=lr_shape, optimizer=optimizer, loss=loss)

    loss_file = open(model_save_dir + 'losses.txt', 'w+')
    loss_file.close()

    discriminator_loss = 0
    gan_loss = 0

    print('-'*15+'> loading model successfully and begin to train......')

    for e in range(epochs):
        print('-'*10, 'Epoch {}'.format(e), '-'*10)
        for _ in tqdm(range(batch_count)):
            # TODO: to figure out the training process of the gan network
            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)

            img_batch_hr = x_train_hr[rand_nums]
            img_batch_lr = x_train_lr[rand_nums]
            generated_img_sr = generator.predict(img_batch_lr)

            real_data_y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            fake_data_y = np.random.random_sample(batch_size) * 0.2

            discriminator.trainable = True

            d_loss_real = discriminator.train_on_batch(img_batch_hr, real_data_y)
            d_loss_fake = discriminator.train_on_batch(generated_img_sr, fake_data_y)
            discriminator_loss = 0.5 * np.add(d_loss_fake, d_loss_real)

            rand_nums = np.random.randint(0, x_train_hr.shape[0], size=batch_size)
            img_batch_hr = x_train_hr[rand_nums]
            img_batch_lr = x_train_lr[rand_nums]

            gan_y = np.ones(batch_size) - np.random.random_sample(batch_size)*0.2
            discriminator.trainable = False
            gan_loss = gan.train_on_batch(img_batch_lr, [img_batch_hr, gan_y])

        print('discriminator_loss -----> :', discriminator_loss)
        print('gan_loss -----> :', gan_loss)

        with open(model_save_dir+'losses.txt', 'a') as f:
            f.write('epoch %d  :  gan_loss--->[ {} ] ; discriminator_loss--->[ {} ]'.format(e, gan_loss, discriminator_loss))

        # TODO: psnr calc
        if e == 0 or e+1 % 100 == 0:
            print('\n'*5)
            generated_test_img = generator.predict(x_test_lr)
            result = []
            for lr_img, true in zip(generated_test_img, x_test_hr):
                result.append(psnr(srgan_utils.srgan_denormalize(lr_img), srgan_utils.srgan_denormalize(true)))
            print('epoch [{}]: psnr ----> {}'.format(e, np.mean(result)))

        if e+1 % 500 == 0:
            generator.save(model_save_dir + 'gen_model%d.h5' % e)
            discriminator.save(model_save_dir + 'dis_model%d.h5' % e)


if __name__ == '__main__':
    # tf.config.experimental_run_functions_eagerly(True)
    print('test tensorflow is built with cuda ------->', tf.test.is_built_with_cuda())
    print('test tensorflow is built with gpu ------->', tf.test.is_built_with_gpu_support())
    print('test gpu is available ------->', tf.test.is_gpu_available(cuda_only=False, min_cuda_compute_capability=None))

    train()
