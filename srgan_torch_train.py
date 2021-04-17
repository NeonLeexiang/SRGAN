"""
    date:       2021/4/7 4:32 下午
    written by: neonleexiang
"""
import os
from datetime import datetime
from srgan_torch_model import Generator, Discriminator
from srgan_torch_loss import GeneratorLoss
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from srgan_torch_data_utils import TrainDatasets, ValDatasets, display_transform
from torch.autograd import Variable
from math import log10
import torchvision
import pandas as pd
import pytorch_ssim

torch.autograd.set_detect_anomaly(True)


def srgan_torch_train(num_epochs=400, batch_size=4, upscale_factor=4, crop_size=128, train_dir='datasets/train/', val_dir='datasets/val/', model_save_dir='srgan_torch_model_file/'):
    if not os.path.exists(model_save_dir):
        os.mkdir(model_save_dir)
    print('-' * 15 + '> start loading data... ', '[ {} ]'.format(datetime.now()))

    # dataloader and remember pytorch needs to write the dataloader file
    train_set = TrainDatasets(data_dir=train_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    val_set = ValDatasets(data_dir=val_dir, crop_size=crop_size, upscale_factor=upscale_factor)
    # then we need to feed the data into DataLoader
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_set, num_workers=4, batch_size=1, shuffle=False)

    print('-' * 15 + '> data loading......')

    # create our network generator and discriminator
    net_generator = Generator(upscale_factor)
    print('#----> generator parameters:', sum(param.numel() for param in net_generator.parameters()))
    net_discriminator = Discriminator()
    print('#----> discriminator parameters:', sum(param.numel() for param in net_discriminator.parameters()))

    '''
        use vgg19 as generator loss
    '''
    # generator_criterion = GeneratorLoss()
    '''
        because of some bug of GeneratorLoss, we change our criterion into MSELoss 
        when using GeneratorLoss it will cause 
    '''
    generator_criterion = torch.nn.MSELoss()

    '''
        torch.cuda is used to set up and run CUDA operations. 
        It keeps track of the currently selected GPU, 
        and all CUDA tensors you allocate will by default be created on that device. 
        The selected device can be changed with a torch.cuda.device context manager.
        
        Cross-GPU operations are not allowed by default, 
        with the exception of copy_() and other methods with copy-like functionality such as to() and cuda(). 
        Unless you enable peer-to-peer memory access, 
        any attempts to launch ops on tensors spread across different devices will raise an error.


    '''

    if torch.cuda.is_available():
        net_generator.cuda()
        net_discriminator.cuda()
        generator_criterion.cuda()

    # if we do not use the lr it will be set as default learning rate 1e-3
    optimizer_generator = torch.optim.Adam(net_generator.parameters(), lr=0.0001)
    optimizer_discriminator = torch.optim.Adam(net_discriminator.parameters(), lr=0.0001)

    # as a good habit we can use results dict to store our results
    results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

    for epoch in range(1, num_epochs+1):
        '''
        tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm(iterator)。
        '''
        train_bar = tqdm(train_loader)
        running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

        # first we train the generator
        net_generator.train()
        # then we train the discriminator
        net_discriminator.train()

        for data, target in train_bar:
            g_update_first = True
            batch_size = data.size(0)
            running_results['batch_sizes'] += batch_size

            '''
            (1) Update D network: maximize D(x)-1-D(G(z))
            '''
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()

            # generate a fake img (from low resolution img to hr img)
            fake_img = net_generator(z)

            # then use fake img to train the discriminator
            net_discriminator.zero_grad()
            real_out = net_discriminator(real_img).mean()
            fake_out = net_discriminator(fake_img).mean()

            # discriminator loss
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizer_discriminator.step()

            '''
            (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            '''

            # we use MSELoss as the generator_criterion
            net_generator.zero_grad()
            # g_loss = generator_criterion(fake_out, fake_img, real_img)
            g_loss = generator_criterion(fake_img, real_img)
            # g_loss.backward(create_graph=True)

            g_loss.backward()

            fake_img = net_generator(z)
            fake_out = net_discriminator(fake_img).mean()

            optimizer_generator.step()

            # loss for current batch before optimization
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size
            # tqdm.set_description method
            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']))

        net_generator.eval()
        '''
            eval()

            将模型设置成evaluation模式

            仅仅当模型中有Dropout和BatchNorm是才会有影响。

        '''

        out_path = os.path.join(model_save_dir, 'training_results/SRF_' + str(upscale_factor) + '/')
        if not os.path.exists(out_path):
            os.mkdir(out_path)

        if epoch % 200 == 0:
            '''
                torch.no_grad() 是一个上下文管理器，被该语句 wrap 起来的部分将不会track 梯度。
                
                class torch.no_grad[source]

                不能进行梯度计算的上下文管理器。
                当你确定你不调用Tensor.backward()时，不能计算梯度对测试来讲非常有用。
                对计算它将减少内存消耗，否则requires_grad=True。
                在这个模式下，每个计算结果都需要使得requires_grad=False，即使当输入为requires_grad=True。
                当使用enable_grad上下文管理器时这个模式不起作用。这个上下文管理器是线程本地的，对其他线程的计算不起作用。
                同样函数作为一个装饰器(确保用括号实例化。)。
            '''
            with torch.no_grad():
                val_bar = tqdm(val_loader)
                valing_results = {'mse': 0, 'ssims': 0, 'psnr': 0, 'ssim': 0, 'batch_sizes': 0}
                val_images = []
                for val_lr, val_hr_restore, val_hr in val_bar:
                    batch_size = val_lr.size(0)
                    valing_results['batch_sizes'] += batch_size
                    lr = val_lr
                    hr = val_hr
                    # if torch.cuda.is_available():
                    #    lr = lr.cuda()
                    #    hr = hr.cuda()

                    # low resolution img input
                    sr = net_generator(lr)

                    # use mse to get the loss evaluation
                    batch_mse = ((sr - hr) ** 2).data.mean()
                    valing_results['mse'] += batch_mse * batch_size
                    batch_ssim = pytorch_ssim.ssim(sr, hr).item()
                    valing_results['ssims'] += batch_ssim * batch_size

                    # psnr counting
                    valing_results['psnr'] = 10 * log10(
                        (hr.max() ** 2) / (valing_results['mse'] / valing_results['batch_sizes']))
                    valing_results['ssim'] = valing_results['ssims'] / valing_results['batch_sizes']
                    val_bar.set_description(
                        desc='[converting LR images to SR images] PSNR: %.4f dB SSIM: %.4f' % (
                            valing_results['psnr'], valing_results['ssim']))

                    # extend method. val_images is a list object extend method
                    val_images.extend(
                        [display_transform()(val_hr_restore.squeeze(0)), display_transform()(hr.data.cpu().squeeze(0)),
                         display_transform()(sr.data.cpu().squeeze(0))])
                '''
                    torch.stack[source]

                    torch.stack(sequence, dim=0)
                    沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
                    
                    参数:
                    
                    sqequence (Sequence) – 待连接的张量序列
                    dim (int) – 插入的维度。必须介于 0 与 待连接的张量序列数之间。

                '''
                val_images = torch.stack(val_images)
                '''
                    torch.chunk

                    torch.chunk(tensor, chunks, dim=0)
                    在给定维度(轴)上将输入张量进行分块儿。

                    参数:

                    tensor (Tensor) – 待分块的输入张量
                    chunks (int) – 分块的个数
                    dim (int) – 沿着此维度进行分块
                '''
                val_images = torch.chunk(val_images, val_images.size(0) // 15)
                val_save_bar = tqdm(val_images, desc='[saving training results]')
                index = 1
                if epoch % 100 == 0:
                    for image in val_save_bar:
                        '''
                            torchvision.utils.make_grid(tensor, nrow=8, padding=2, 
                                                        normalize=False, range=None, scale_each=False)

                            猜测，用来做 雪碧图的（sprite image）。

                            给定 4D mini-batch Tensor， 形状为 (B x C x H x W),
                            或者一个a list of image，做成一个size为(B / nrow, nrow)的雪碧图。

                            normalize=True ，会将图片的像素值归一化处理
                            如果 range=(min, max)， min和max是数字，那么min，max用来规范化image
                            scale_each=True ，每个图片独立规范化，而不是根据所有图片的像素最大最小值来规范化
                        '''
                        image = torchvision.utils.make_grid(image, nrow=3, padding=5)
                        torchvision.utils.save_image(image, os.path.join(out_path, 'epoch_%d_index_%d.png' % (epoch, index)), padding=5)
                        index += 1

            # save model parameters
            # if epoch % 200 == 0:
            torch.save(net_generator.state_dict(), os.path.join(model_save_dir, 'epochs/netG_epoch_%d_%d.pth' % (upscale_factor, epoch)))
            torch.save(net_discriminator.state_dict(), os.path.join(model_save_dir, 'epochs/netD_epoch_%d_%d.pth' % (upscale_factor, epoch)))
            # save loss\scores\psnr\ssim
            results['d_loss'].append(running_results['d_loss'] / running_results['batch_sizes'])
            results['g_loss'].append(running_results['g_loss'] / running_results['batch_sizes'])
            results['d_score'].append(running_results['d_score'] / running_results['batch_sizes'])
            results['g_score'].append(running_results['g_score'] / running_results['batch_sizes'])
            results['psnr'].append(valing_results['psnr'])
            results['ssim'].append(valing_results['ssim'])

            # if epoch % 200 == 0:
            out_path = os.path.join(model_save_dir, 'statistics/')
            data_frame = pd.DataFrame(
                data={'Loss_D': results['d_loss'], 'Loss_G': results['g_loss'], 'Score_D': results['d_score'],
                      'Score_G': results['g_score'], 'PSNR': results['psnr'], 'SSIM': results['ssim']},
                index=range(1, epoch + 1))
            if not os.path.exists(os.path.join(out_path, 'srf_'+str(upscale_factor))):
                os.mkdir(os.path.join(out_path, 'srf_'+str(upscale_factor)))
            data_frame.to_csv(out_path + 'srf_' + str(upscale_factor) + '_train_results.csv', index_label='Epoch')


if __name__ == '__main__':
    srgan_torch_train()




