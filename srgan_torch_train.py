"""
    date:       2021/4/7 4:32 下午
    written by: neonleexiang
"""
import srgan_utils
from datetime import datetime
from srgan_torch_model import Generator, Discriminator
from srgan_torch_loss import GeneratorLoss
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


def srgan_torch_train(num_epochs=50, upscale_factor=4, train_dir='datasets/train/', val_dir='datasets/val/', model_save_dir='srgan_model_file/'):
    print('-' * 15 + '> start loading data... ', '[ {} ]'.format(datetime.now()))
    x_train_lr, x_train_hr, x_test_lr, x_test_hr = srgan_utils.load_train_val_data(train_dir, val_dir)

    # x_train_lr = x_train_lr[:4]
    # x_train_hr = x_train_hr[:4]
    # x_test_lr = x_test_lr[:2]
    # x_test_hr = x_test_hr[:2]

    print('-' * 15 + '> data loading......')
    print(x_train_lr.shape)
    print(x_train_hr.shape)
    print(x_test_lr.shape)
    print(x_test_hr.shape)

    net_generator = Generator(upscale_factor)
    print('#----> generator parameters:', sum(param.numel() for param in net_generator.parameters()))
    net_discriminator = Discriminator()
    print('#----> discriminator parameters:', sum(param.numel() for param in net_discriminator.parameters()))

    # TODO: to figure out what is GeneratorLoss and its usage
    generator_criterion = GeneratorLoss()

    if torch.cuda.is_available():
        # TODO: figure out the usage of cuda method
        net_generator.cuda()
        net_discriminator.cuda()
        generator_criterion.cuda()

    optimizer_generator = torch.optim.Adam(net_generator.parameters())
    optimizer_discriminator = torch.optim.Adam(net_discriminator.parameters())

    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, num_epochs+1):
        train_bar = tqdm()


