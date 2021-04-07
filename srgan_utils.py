"""
    date:       2021/4/3 11:08 上午
    written by: neonleexiang
"""
import numpy as np
import cv2 as cv
import os


def srgan_normalize(input_data):
    return (input_data.astype(np.float32) - 127.5) / 127.5


def srgan_denormalize(input_data):
    input_data = (input_data + 1) * 127.5
    return input_data.astype(np.uint8)


def hr_images(images):
    return np.array(images)


def lr_images(images, downscale):
    res_images = []
    for i in range(len(images)):
        res_images.append(cv.resize(images[i],
                                    (images[i].shape[1] // downscale, images[i].shape[0] // downscale),
                                    interpolation=cv.INTER_CUBIC))
    return np.array(res_images)


def load_img_list(data_path):
    res_lst = []
    for img_name in os.listdir(data_path):
        img = cv.imread(os.path.join(data_path, img_name))
        res_lst.append(img)
    return np.array(res_lst)


def load_train_val_data(train_data_path, val_data_path):
    x_train_data = load_img_list(train_data_path)
    x_test_data = load_img_list(val_data_path)

    x_train_hr = srgan_normalize(hr_images(x_train_data))
    x_train_lr = srgan_normalize(lr_images(x_train_data, 4))

    x_test_hr = srgan_normalize(hr_images(x_test_data))
    x_test_lr = srgan_normalize(lr_images(x_test_data, 4))

    return x_train_lr, x_train_hr, x_test_lr, x_test_hr


if __name__ == '__main__':
    train_lr, train_hr, test_lr, test_hr = load_train_val_data('datasets/train/', 'datasets/val/')
    print(train_lr.shape)
    cv.imshow('train_lr', srgan_denormalize(train_lr[1]))
    print(train_hr.shape)
    cv.imshow('train_hr', srgan_denormalize(train_hr[1]))
    print(test_lr.shape)
    print(test_hr.shape)
    cv.waitKey(0)
    cv.destroyAllWindows()







