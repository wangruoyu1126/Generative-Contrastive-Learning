
########################
# Toy Example on MNIST #
########################

# import torch
# import numpy as np
# from torchvision import datasets
# import torchvision.transforms as transforms
# from torchvision.utils import save_image
# from model import ConvDenoiser
# import torch.nn as nn
#
#
# transform = transforms.ToTensor()
# train_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=True,
#                                    download=True, transform=transform)
# test_data = datasets.MNIST(root='~/.pytorch/MNIST_data/', train=False,
#                                   download=True, transform=transform)
#
# train_loader = torch.utils.data.DataLoader(train_data, batch_size=256, num_workers=16)
# test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, num_workers=16)
#
#
# model = ConvDenoiser().cuda()
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#
# # number of epochs to train the model
# n_epochs = 50
#
# # for adding noise to images
# noise_factor = 0.5
#
# for epoch in range(1, n_epochs + 1):
#     train_loss = 0.0
#     batch_num = 0
#     for data in train_loader:
#         images, _ = data
#
#         noisy_imgs = images + noise_factor * torch.randn(*images.shape)
#         noisy_imgs = np.clip(noisy_imgs, 0., 1.)
#
#         noisy_imgs = noisy_imgs.cuda(non_blocking=True)
#         images = images.cuda(non_blocking=True)
#
#         optimizer.zero_grad()
#         outputs = model(noisy_imgs)
#
#         if batch_num == 0:
#             save_image(outputs, 'results/generated_samples/gen_epoch{}_batch{}.png'.format(epoch, batch_num))
#             save_image(noisy_imgs, 'results/generated_samples/input_epoch{}_batch{}.png'.format(epoch, batch_num))
#
#         loss = criterion(outputs, images)
#
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item() * images.size(0)
#
#         batch_num += 1
#
#     # print avg training statistics
#     train_loss = train_loss / len(train_loader)
#     print('Epoch: {} \tTraining Loss: {:.6f}'.format(
#         epoch,
#         train_loss
#     ))
#
#
# assert 1==0
#




###########################
# Load my CIFAR10 dataset #
###########################
import torch
import numpy as np
from torchvision import datasets
import torchvision.transforms as T
from torchvision.utils import save_image
from model import ConvDenoiser, GeneratorUNet, Generator, GeneratorResNet
import torch.nn as nn
import utils
from torch.utils.data import DataLoader
from tqdm import tqdm

dataset = utils.CIFAR10LabelFolder(transform=utils.train_transform, train=True)
dataloader = DataLoader(dataset, batch_size=256,
                        shuffle=True, num_workers=16,
                        pin_memory=True, drop_last=True)

model = GeneratorResNet(input_shape=(3,32,32), num_residual_blocks=0).cuda()

# for name, module in model.named_children():
#     print(name, module)
# assert 1==0
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)

# number of epochs to train the model
n_epochs = 50

for epoch in range(1, n_epochs + 1):
    train_loss = 0.0
    batch_num = 0
    for _, _, org_img, _, pos_gen, _, _ in tqdm(dataloader):
        dirty, clean = org_img.cuda(non_blocking=True), pos_gen.cuda(non_blocking=True)

        # print('img_size', dirty.shape)
        # transform_resize = T.Resize(512)
        # dirty = transform_resize(dirty)
        # clean = transform_resize(clean)
        # print('img_size', dirty.shape)

        output = model(dirty)
        if batch_num == 0:
            save_image(output, 'results/generated_samples/gen_epoch{}_batch{}.png'.format(epoch, batch_num))
            # save_image(dirty, 'results/generated_samples/dirty_epoch{}_batch{}.png'.format(epoch, batch_num))
            save_image(clean, 'results/generated_samples/clean_{}_batch{}.png'.format(epoch, batch_num))

        loss = criterion(output, clean)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_num += 1

    print("======> epoch: {}/{}, Loss:{}".format(epoch, n_epochs, loss.item()))


torch.save(model.state_dict(), 'results/pretrain_generator.pth')












