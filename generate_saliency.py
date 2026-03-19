# from torch.utils.data import DataLoader
# import utils
# from tqdm import tqdm
# import torch
# from saliency import *
# import shutil
#
# batch_size = 1
#
# utils.set_seed(1234)

# CIFAR
# train_data = utils.CIFAR100Pair_Org(root='data', train=True, transform=utils.train_transform)
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
#                           drop_last=True)
# test_data = utils.CIFAR100Pair_Org(root='data', train=False, transform=utils.test_transform)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# STL
# train_data = utils.STL10Pair_Org(root='data', split='train+unlabeled', transform=utils.train_transform)
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True,
#                                   drop_last=True)
# test_data = utils.STL10Pair_Org(root='data', split='test', transform=utils.test_transform)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

# VOC Detection
# train_data = utils.VOCDec_Org(root='data', image_set='train')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)
# test_data = utils.VOCDec_Org(root='data', image_set='val')
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)


# VOC Segmentation
# train_data = utils.VOCSeg_Org(root='data', image_set='train')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)
# test_data = utils.VOCSeg_Org(root='data', image_set='val')
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)

# CelebA
# train_data = utils.CelebA_Org(root='data', split='train', target_type='bbox')
# train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)
# test_data = utils.CelebA_Org(root='data', split='test', target_type='bbox')
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=False,
#                           drop_last=True)





# print('handling train data')
# i = 0
# target_lst = []
# for org_img, target in tqdm(train_loader):
#     # print('i', i)
#     # print(org_img.shape)
#     # print(target)
#
#     sal_org_img = get_bin_saliency_batch(org_img)
#     sal_org_img = torch.from_numpy(np.stack(sal_org_img, axis=0))
#
#     # print(sal_org_img.shape)
#     mask_org_img = sal_org_img.unsqueeze(dim=3).expand(-1, -1, -1, 3)
#     # print(mask_org_img.shape)
#
#     pos_gen = org_img.clone()
#     neg_gen = org_img.clone()
#     pos_gen[mask_org_img == False] = 0
#     neg_gen[mask_org_img] = 0
#     # print(pos_gen.shape)
#     # print(neg_gen.shape)
#
#     # org_img = torch.permute(org_img, (0, 2, 3, 1))
#     # print(org_img[0].shape)
#
#     cv2.imwrite('data/CelebA/train/img/{}.jpg'.format(i), org_img[0].detach().numpy().astype('uint8'))
#     cv2.imwrite('data/CelebA/train/mask/{}.jpg'.format(i), 255 * mask_org_img[0].detach().numpy().astype('uint8'))
#     cv2.imwrite('data/CelebA/train/pos/{}.jpg'.format(i), pos_gen[0].detach().numpy().astype('uint8'))
#     cv2.imwrite('data/CelebA/train/neg/{}.jpg'.format(i), neg_gen[0].detach().numpy().astype('uint8'))
#
#     target_lst.append(target[0].tolist())
#     # print(target_lst)
#     # shutil.copyfile(target_file[0], 'data/VOC_Seg/train/targets/{}.png'.format(i))
#
#     # assert i!=5
#
#     i += 1
#
# with open('data/CelebA/train/labels.txt', 'w') as f:
#     for line in target_lst:
#         f.write(f"{line}\n")














###################
# NICO #
###################

import os
from torchvision import transforms
import torch
from saliency import *

mode = 'train'

img_num = len(os.listdir('data/Animal/{}/img'.format(mode)))
# convert_tensor = transforms.ToTensor()

for i in range(img_num):
    print('{} out of {}'.format(i, img_num))
    img_path = 'data/Animal/{}/img/{}.jpg'.format(mode, i)
    org_img = cv2.imread(img_path)
    org_img = np.expand_dims(org_img, axis=0)

    sal_org_img = get_bin_saliency_batch(org_img)
    sal_org_img = torch.from_numpy(np.stack(sal_org_img, axis=0))

    mask_org_img = sal_org_img.unsqueeze(dim=3).expand(-1, -1, -1, 3)
    pos_gen = org_img
    pos_gen[mask_org_img == False] = 0

    cv2.imwrite('data/Animal/{}/pos/{}.jpg'.format(mode, i), pos_gen[0].astype('uint8'))



mode = 'test'

img_num = len(os.listdir('data/Animal/{}/img'.format(mode)))
# convert_tensor = transforms.ToTensor()

for i in range(img_num):
    print('{} out of {}'.format(i, img_num))
    img_path = 'data/Animal/{}/img/{}.jpg'.format(mode, i)
    org_img = cv2.imread(img_path)
    org_img = np.expand_dims(org_img, axis=0)

    sal_org_img = get_bin_saliency_batch(org_img)
    sal_org_img = torch.from_numpy(np.stack(sal_org_img, axis=0))

    mask_org_img = sal_org_img.unsqueeze(dim=3).expand(-1, -1, -1, 3)
    pos_gen = org_img
    pos_gen[mask_org_img == False] = 0

    cv2.imwrite('data/Animal/{}/pos/{}.jpg'.format(mode, i), pos_gen[0].astype('uint8'))






































