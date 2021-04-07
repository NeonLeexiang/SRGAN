"""
    date:       2021/4/3 2:03 下午
    written by: neonleexiang
"""
import cv2 as cv
import os


def single_img_cut(prefix_path, img_name, saving_path, size=256):
    img = cv.imread(os.path.join(prefix_path, img_name))
    # we cut the central part of the img
    height, weight = (img.shape[0] - 256) // 2, (img.shape[1] - 256) // 2
    img_cut = img[height: height+256, weight: weight+256]
    cv.imwrite(os.path.join(saving_path, img_name), img_cut)
    print('-----> img:', img_name, 'saving successfully......')


def dir_img_cut(data_path, saving_path, size=256):
    print('cutting datasets --->', data_path)
    if not os.path.exists(saving_path):
        os.mkdir(saving_path)
    for img_name in os.listdir(data_path):
        print('the name of img ----->', img_name)
        single_img_cut(data_path, img_name, saving_path, size)
    print('########### end of cutting --->', data_path)


if __name__ == '__main__':
    dir_img_cut('images/train/', 'datasets/train/')
    dir_img_cut('images/test/', 'datasets/test/')
    dir_img_cut('images/val/', 'datasets/val/')


