from PIL import Image
from torchvision import transforms
from torchvision.datasets import STL10, CIFAR10, CIFAR100
from torchvision.datasets import VOCDetection, VOCSegmentation, CelebA

from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
import random
from skimage import transform

np.random.seed(0)


class STL10Pair(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target

class STL10Pair_Index(STL10):
    def __getitem__(self, index):
        img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, target, index


class STL10Pair_Org(STL10):
    def __getitem__(self, index):
        org_img, target = self.data[index], self.labels[index]
        img = Image.fromarray(np.transpose(org_img, (1, 2, 0)))

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        return pos_1, pos_2, np.transpose(org_img, (1, 2, 0)), target


class CIFAR10Pair(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = test_transform(img)

        return pos_1, pos_2, img, target


class CIFAR10Pair_Org(CIFAR10):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        org_img, target = self.data[index], self.targets[index]
        img = Image.fromarray(org_img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, org_img, target

class CIFAR10Pair_Index(CIFAR10):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index

class CIFAR100Pair(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target

class CIFAR100Pair_Org(CIFAR100):
    """CIFAR10 Dataset.
    """
    def __getitem__(self, index):
        org_img, target = self.data[index], self.targets[index]
        img = Image.fromarray(org_img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, org_img, target


class CIFAR100Pair_Index(CIFAR100):
    """CIFAR10 Dataset.
    """

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return pos_1, pos_2, target, index


###############
# New Dataset #
###############
from xml.etree.ElementTree import parse as ET_parse

# class VOCDec_Org(VOCDetection):
#     def __getitem__(self, index):
#         img = Image.open(self.images[index]).convert("RGB")
#         org_img = np.asarray(img)
#         target_file = self.annotations[index]
#         # target = self.parse_voc_xml(ET_parse(self.annotations[index]).getroot())
#
#         return org_img, target_file
#
#
# class VOCSeg_Org(VOCSegmentation):
#     def __getitem__(self, index):
#         img = Image.open(self.images[index]).convert("RGB")
#         org_img = np.asarray(img)
#         target_file = self.masks[index]
#         # target = Image.open(self.masks[index])
#
#         return org_img, target_file
#
# import PIL
# class CelebA_Org(CelebA):
#     def __getitem__(self, index):
#         X = PIL.Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", self.filename[index]))
#         org_img = np.asarray(X)
#
#         target = []
#         target.append(self.bbox[index, :])
#
#         if target:
#             target = tuple(target) if len(target) > 1 else target[0]
#         else:
#             target = None
#
#         return org_img, target





# SEED
def set_seed(seed):
    if_cuda = torch.cuda.is_available()
    torch.manual_seed(seed)
    if if_cuda:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_log(print_str, log_file, print_=False):
    if print_:
        print(print_str)
    if log_file is None:
        return
    with open(log_file, 'a') as f:
        f.write('\n')
        f.write(print_str)


class GaussianBlur(object):
    # Implements Gaussian blur as described in the SimCLR paper
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        # kernel size is set to be 10% of the image height/width
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)

        # blur the image with a 50% chance
        prob = np.random.random_sample()

        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)

        return sample


# just follow the previous work -- DCL, NeurIPS2020
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlur(kernel_size=int(0.1 * 32)),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

sal_transform = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

org_img_transform = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

test_transform = transforms.Compose([
    # transforms.CenterCrop(224),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])




#########################
# Load Data from Folder #
#########################

class CIFAR10Folder(Dataset):
    """
    Custom Dataset to load original img, positive generation and negative generation
    """
    def __init__(self, transform, train):
        if train:
            data_root = 'data/CIFAR10/train'
            data_len = 50000
        else:
            data_root = 'data/CIFAR10/test'
            data_len = 10000
        img_root = data_root + '/img'
        # mask_root = data_root + '/mask'
        pos_root = data_root + '/pos'
        # neg_root = data_root + '/neg'
        self.imgs = []
        # self.masks = []
        self.poss = []
        # self.negs = []
        self.transform = transform
        targets_file = open(data_root + '/labels.txt', 'r')
        self.targets = [int(item) for item in targets_file.read().split('\n')]

        for i in range(data_len):
            filename = '{}.jpg'.format(i)
            img = cv2.imread(img_root+'/' + filename)
            # mask = cv2.imread(mask_root + '/' + filename)
            pos = cv2.imread(pos_root + '/' + filename)
            # neg = cv2.imread(neg_root + '/' + filename)
            self.imgs.append(img)
            # self.masks.append(mask)
            self.poss.append(pos)
            # self.negs.append(neg)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        # img, mask, pos_gen, neg_gen = self.imgs[index], self.masks[index], self.poss[index], self.negs[index]
        img, pos_gen = self.imgs[index], self.poss[index]

        target = int(self.targets[index])

        img = Image.fromarray(img)
        # mask = Image.fromarray(mask)
        pos_gen = Image.fromarray(pos_gen)
        # neg_gen = Image.fromarray(neg_gen)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)

        img = org_img_transform(img)
        # mask = norm_transform(mask)
        pos_gen = sal_transform(pos_gen)
        # neg_gen = norm_transform(neg_gen)

        return pos_1, pos_2, img, pos_gen, target



class CIFAR100Folder(Dataset):
    """
    Custom Dataset to load original img, positive generation and negative generation
    """
    def __init__(self, transform, train):
        if train:
            data_root = 'data/CIFAR100/train'
            data_len = 50000
        else:
            data_root = 'data/CIFAR100/test'
            data_len = 10000
        img_root = data_root + '/img'
        pos_root = data_root + '/pos'
        self.imgs = []
        self.poss = []
        self.transform = transform
        targets_file = open(data_root + '/labels.txt', 'r')
        self.targets = [int(item) for item in targets_file.read().split('\n')]

        for i in range(data_len):
            filename = '{}.jpg'.format(i)
            img = cv2.imread(img_root+'/' + filename)
            pos = cv2.imread(pos_root + '/' + filename)
            self.imgs.append(img)
            self.poss.append(pos)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, pos_gen = self.imgs[index], self.poss[index]

        target = int(self.targets[index])

        img = Image.fromarray(img)
        pos_gen = Image.fromarray(pos_gen)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)

        img = org_img_transform(img)
        pos_gen = sal_transform(pos_gen)

        return pos_1, pos_2, img, pos_gen, target



class STL10Folder(Dataset):
    """
    Custom Dataset to load original img, positive generation and negative generation
    """
    def __init__(self, transform, train):
        if train:
            data_root = 'data/STL10/train'
            data_len = 105000
        else:
            data_root = 'data/STL10/test'
            data_len = 8000
        img_root = data_root + '/img'
        pos_root = data_root + '/pos'
        self.imgs = []
        self.poss = []
        self.transform = transform
        targets_file = open(data_root + '/labels.txt', 'r')
        self.targets = [int(item) for item in targets_file.read().split('\n')]

        for i in range(data_len):
            filename = '{}.jpg'.format(i)
            img = cv2.imread(img_root+'/' + filename)
            pos = cv2.imread(pos_root + '/' + filename)
            self.imgs.append(img)
            self.poss.append(pos)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, pos_gen = self.imgs[index], self.poss[index]

        target = int(self.targets[index])

        img = Image.fromarray(img)
        pos_gen = Image.fromarray(pos_gen)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)

        img = org_img_transform(img)
        pos_gen = sal_transform(pos_gen)

        return pos_1, pos_2, img, pos_gen, target





class NICODataset(Dataset):
    """
    Custom Dataset to load original img, positive generation and negative generation
    """
    def __init__(self, transform, train):
        if train:
            data_root = 'data/Animal/train'
        else:
            data_root = 'data/Animal/test'

        img_root = data_root + '/img'
        pos_root = data_root + '/pos'
        self.imgs = []
        self.poss = []
        self.transform = transform
        targets_file = open(data_root + '/labels.txt', 'r')
        self.targets = [int(item) for item in targets_file.read().split('\n')]

        data_len = len(os.listdir(img_root))

        for i in range(data_len):
            filename = '{}.jpg'.format(i)
            img = cv2.imread(img_root + '/' + filename)
            pos = cv2.imread(pos_root + '/' + filename)
            self.imgs.append(img)
            self.poss.append(pos)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img, pos_gen = self.imgs[index], self.poss[index]
        target = int(self.targets[index])

        img = Image.fromarray(img)
        pos_gen = Image.fromarray(pos_gen)

        if self.transform is not None:
            pos_1 = self.transform(img)
            pos_2 = self.transform(img)
            pos_3 = self.transform(img)

        img = test_transform(img)
        pos_gen = test_transform(pos_gen)

        return pos_1, pos_2, pos_3, img, pos_gen, target






if __name__ == '__main__':
    from torch.utils.data import DataLoader
    from tqdm import tqdm
    from torchvision.utils import save_image
    dataset = CIFAR10Folder(transform=train_transform)
    dataloader = DataLoader(dataset, batch_size=4,
                            shuffle=True, num_workers=16,
                            pin_memory=True, drop_last=True)

    for pos_1, pos_2, org_img, mask, pos_gen, neg_gen in tqdm(dataloader):
        print('pos_1', pos_1.shape)
        print('pos_2', pos_2.shape)
        print('org_img', org_img.shape)
        print('mask', mask.shape)
        print('pos_gen', pos_gen.shape)
        print('neg_gen', neg_gen.shape)

        save_image(pos_1, 'pos_1.jpg')
        save_image(pos_2, 'pos_2.jpg')
        save_image(org_img, 'org_img.jpg')
        save_image(mask, 'mask.jpg')
        save_image(pos_gen, 'pos_gen.jpg')
        save_image(neg_gen, 'neg_gen.jpg')
        # cv2.imwrite('org_img.jpg', org_img[0].numpy())
        # cv2.imwrite('mask.jpg', mask[0].numpy())
        # cv2.imwrite('pos_gen.jpg', pos_gen[0].numpy())
        # cv2.imwrite('neg_gen.jpg', neg_gen[0].numpy())
        assert 1 == 0




















