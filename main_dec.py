import argparse
import os
from datetime import datetime
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,4,5,7'
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F

import utils_dec as utils
from saliency import *
from model import Model, PretrainedResNet50
import random

from torchvision.utils import save_image, draw_segmentation_masks
from torchvision import transforms


def get_negative_mask(batch_size):
    negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)
    for i in range(batch_size):
        negative_mask[i, i] = 0
        negative_mask[i, i + batch_size] = 0

    negative_mask = torch.cat((negative_mask, negative_mask), 0)
    return negative_mask


def train(net, data_loader, train_optimizer, temperature, debiased, tau_plus):
    # baseline train generator
    print('doing original training')
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    for pos_1, pos_2, _, _, _ in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        # print('pos_1 shape', pos_1.shape)
        # print('pos_2 shape', pos_2.shape)
        # save_image(pos_1, 'pos_1.png')
        # save_image(pos_2, 'pos_2.png')
        # assert 1==0
        feature_1, out_1 = net(pos_1)
        feature_2, out_2 = net(pos_2)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)


        # # generated neg score
        # gen_pos_1 = torch.exp(torch.sum(out_1 * out_3, dim=-1) / temperature)
        # gen_pos_2 = torch.exp(torch.sum(out_2 * out_3, dim=-1) / temperature)
        # gen_Ps = torch.cat([gen_pos_1, gen_pos_2], dim=0)

        # End calculating negative generation

        # contrastive loss - InfoNCE + generated negative loss
        # loss = (- torch.log((pos + gen_Ps) / (pos + Ng))).mean()


        # contrastive loss - InfoNCE
        loss = (- torch.log(pos / (pos + Ng) )).mean()

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))

    return total_loss / total_num


# def train_generator(enc, gen,
#                     data_loader, opt_gen,
#                     temperature, gen_weight):
#     print('doing train generator')
#     gen.train()
#     enc.eval()
#     total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
#     batch_num = 0
#
#     # get/generate positive samples
#     for pos_1, pos_2, org_img, _, _, _, _ in train_bar:
#         pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
#         org_img = org_img.cuda(non_blocking=True)
#
#         with torch.no_grad():
#             feature_1, out_1 = enc(pos_1)
#             feature_2, out_2 = enc(pos_2)
#
#             # neg score
#             out = torch.cat([out_1, out_2], dim=0)
#             neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#             mask = get_negative_mask(batch_size).cuda()
#             neg = neg.masked_select(mask).view(2 * batch_size, -1)
#
#             # pos score
#             pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#             pos = torch.cat([pos, pos], dim=0)
#
#             # estimator g()
#             Ng = neg.sum(dim=-1)
#
#         # Add generated samples
#         pos_gen = gen(org_img)
#
#         # print("pos_gen shape", pos_gen.shape)
#         if batch_num % 100 == 0:
#             save_image(pos_gen, 'results/generated_samples/epoch{}_batch{}_gen.png'.format(epoch, batch_num))
#
#         # features from encoder
#         # with torch.no_grad():
#         feature_pos_gen, out_pos_gen = enc(pos_gen)
#
#         # generated neg score
#         gen_pos_1 = torch.exp(torch.sum(out_1 * out_pos_gen, dim=-1) / temperature)
#         gen_pos_2 = torch.exp(torch.sum(out_2 * out_pos_gen, dim=-1) / temperature)
#         gen_Ps = torch.cat([gen_pos_1, gen_pos_2], dim=0)
#
#         # End calculating negative generation
#
#         # contrastive loss - InfoNCE + generated negative loss
#         loss = (torch.log((pos + gen_weight * gen_Ps) / (pos + Ng))).mean()
#         # loss = (- torch.log(pos / (pos + Ng))).mean()
#
#         opt_gen.zero_grad()
#         loss.backward()
#         opt_gen.step()
#
#         total_num += batch_size
#         total_loss += loss.item() * batch_size
#
#         train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
#         batch_num += 1
#     return total_loss / total_num


# Use generator to generate 1 positive sample, only train encoder
# def train_encoder(gen, enc,
#                   data_loader, opt_enc,
#                   temperature, gen_weight):
#     print('doing train encoder')
#     enc.train()
#     gen.eval()
#
#     total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
#     batch_num = 0
#
#     for pos_1, pos_2, org_img, _, _, _ in train_bar:
#         pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
#         org_img = org_img.cuda(non_blocking=True)
#         feature_1, out_1 = enc(pos_1)
#         feature_2, out_2 = enc(pos_2)
#         # feature_org_img, out_org_img = enc(org_img)
#
#         # neg score
#         out = torch.cat([out_1, out_2], dim=0)
#         neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
#         mask = get_negative_mask(batch_size).cuda()
#         neg = neg.masked_select(mask).view(2 * batch_size, -1)
#
#         # pos score
#         pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
#         pos = torch.cat([pos, pos], dim=0)
#
#         # estimator g()
#         Ng = neg.sum(dim=-1)
#
#         # Add generated samples
#         # with torch.no_grad():
#         pos_gen = gen(org_img)
#
#         # print("pos_gen shape", pos_gen.shape)
#         if batch_num % 100 == 0:
#             save_image(pos_gen, 'results/generated_samples/epoch{}_batch{}_enc.png'.format(epoch, batch_num))
#
#         # features from encoder
#         # with torch.no_grad():
#         feature_pos_gen, out_pos_gen = enc(pos_gen)
#
#         # generated neg score
#         gen_pos_1 = torch.exp(torch.sum(out_1 * out_pos_gen, dim=-1) / temperature)
#         gen_pos_2 = torch.exp(torch.sum(out_2 * out_pos_gen, dim=-1) / temperature)
#         gen_Ps = torch.cat([gen_pos_1, gen_pos_2], dim=0)
#
#         # End calculating negative generation
#
#         # contrastive loss - InfoNCE + generated negative loss
#         loss = (- torch.log((pos + gen_weight * gen_Ps) / (pos + Ng))).mean()
#         # loss = (- torch.log(pos / (pos + Ng))).mean()
#
#         opt_enc.zero_grad()
#         loss.backward()
#         opt_enc.step()
#
#         total_num += batch_size
#         total_loss += loss.item() * batch_size
#
#         train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
#         batch_num += 1
#     return total_loss / total_num

# use traditional saliency method, add salient object as positive sample
def train_encoder(enc,
                  data_loader, opt_enc,
                  temperature, pos_gen_weight,
                  debiased, tau_plus):
    print('doing train encoder')
    enc.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    batch_num = 0
    for pos_1, pos_2, _, pos_gen, _ in train_bar:
        pos_1, pos_2 = pos_1.cuda(non_blocking=True), pos_2.cuda(non_blocking=True)
        pos_gen = pos_gen.cuda(non_blocking=True)

        feature_1, out_1 = enc(pos_1)
        feature_2, out_2 = enc(pos_2)

        # neg score
        out = torch.cat([out_1, out_2], dim=0)
        neg = torch.exp(torch.mm(out, out.t().contiguous()) / temperature)
        mask = get_negative_mask(batch_size).cuda()
        neg = neg.masked_select(mask).view(2 * batch_size, -1)

        # pos score
        pos = torch.exp(torch.sum(out_1 * out_2, dim=-1) / temperature)
        pos = torch.cat([pos, pos], dim=0)

        # estimator g()
        if debiased:
            N = batch_size * 2 - 2
            Ng = (-tau_plus * N * pos + neg.sum(dim = -1)) / (1 - tau_plus)
            # constrain (optional)
            Ng = torch.clamp(Ng, min = N * np.e**(-1 / temperature))
        else:
            Ng = neg.sum(dim=-1)

        #########################
        # Add generated samples #
        #########################

        feature_pos_gen_1, out_pos_gen_1 = enc(pos_gen)

        # generated pos score
        gen_pos_1 = torch.exp(torch.sum(out_1 * out_pos_gen_1, dim=-1) / temperature)
        gen_pos_2 = torch.exp(torch.sum(out_2 * out_pos_gen_1, dim=-1) / temperature)
        gen_Ps = torch.cat([gen_pos_1, gen_pos_2], dim=0)
        # End calculating negative generation

        # contrastive loss - InfoNCE + generated negative loss
        # loss = (- torch.log((pos + pos_gen_weight * gen_Ps) / (pos + Ng + neg_gen_weight * gen_Ng))).mean()
        loss = (- torch.log((pos + pos_gen_weight * gen_Ps) / (pos + Ng))).mean()
        # loss = (- torch.log((pos) / (pos + Ng))).mean()

        opt_enc.zero_grad()
        loss.backward()
        opt_enc.step()

        total_num += batch_size
        total_loss += loss.item() * batch_size

        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.4f}'.format(epoch, epochs, total_loss / total_num))
        batch_num += 1
    return total_loss / total_num




# def test(net, memory_data_loader, test_data_loader):
#     net.eval()
#     total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
#     with torch.no_grad():
#         # generate feature bank
#         for data, _, target in tqdm(memory_data_loader, desc='Feature extracting'):
#             feature, out = net(data.cuda(non_blocking=True))
#             feature_bank.append(feature)
#         # [D, N]
#         feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
#         # [N]
#         try:
#             feature_labels = torch.tensor(memory_data_loader.dataset.labels, device=feature_bank.device, dtype=torch.int64)
#         except:
#             feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device, dtype=torch.int64)
#
#         # loop test data to predict the label by weighted knn search
#         test_bar = tqdm(test_data_loader)
#         for data, _, _, _, target in test_bar:
#             # print('test - data shape is', data.shape)
#             data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
#             feature, out = net(data)
#
#             total_num += data.size(0)
#             # compute cos similarity between each feature vector and feature bank ---> [B, N]
#             sim_matrix = torch.mm(feature, feature_bank)
#
#             # [B, K]
#             sim_weight, sim_indices = sim_matrix.topk(k=k, dim=-1)
#
#             # [B, K]
#             sim_labels = torch.gather(feature_labels.expand(data.size(0), -1), dim=-1, index=sim_indices)
#             sim_weight = (sim_weight / temperature).exp()
#
#             # counts for each class
#             one_hot_label = torch.zeros(data.size(0) * k, c, device=sim_labels.device)
#             # [B*K, C]
#             one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
#             # weighted score ---> [B, C]
#             pred_scores = torch.sum(one_hot_label.view(data.size(0), -1, c) * sim_weight.unsqueeze(dim=-1), dim=1)
#
#             pred_labels = pred_scores.argsort(dim=-1, descending=True)
#
#             total_top1 += torch.sum((pred_labels[:, :1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
#             total_top5 += torch.sum((pred_labels[:, :5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
#             test_bar.set_description('Test Epoch: [{}/{}] Acc@1:{:.2f}% Acc@5:{:.2f}%'
#                                      .format(epoch, epochs, total_top1 / total_num * 100, total_top5 / total_num * 100))
#
#     return total_top1 / total_num * 100, total_top5 / total_num * 100


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train SimCLR')
    parser.add_argument('--feature_dim', default=128, type=int, help='Feature dim for latent vector')
    parser.add_argument('--temperature', default=0.5, type=float, help='Temperature used in softmax')
    parser.add_argument('--tau_plus', default=0.1, type=float, help='Positive class prior')
    parser.add_argument('--k', default=200, type=int, help='Top k most similar images used to predict the label')
    parser.add_argument('--batch_size', default=256, type=int, help='Number of images in each mini-batch')
    parser.add_argument('--epochs', default=200, type=int, help='Number of sweeps over the dataset to train')
    parser.add_argument('--debiased', default=False, type=bool, help='Debiased contrastive loss or standard loss')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='experiment dataset')
    parser.add_argument('--name', type=str, default='None', help='experiment name')
    parser.add_argument('--pretrain_model', default=None, type=str, help='pretrain model used?')
    parser.add_argument('--baseline', action="store_true", default=False, help='SSL baseline?')

    # Ruoyu - new argument
    parser.add_argument('--pos_gen_weight', default=1, type=float, help='weight of generated positive sample in Loss')
    parser.add_argument('--neg_gen_weight', default=0, type=float, help='weight of generated negative sample in Loss')
    parser.add_argument('--lr_enc', default=1e-3, type=float, help='lr for encoder')
    parser.add_argument('--lr_gen', default=1e-3, type=float, help='lr for generator')
    parser.add_argument('--pretrained_backbone', action="store_true", default=False, help='Use Pretrained ResNet50 as backbone?')

    # args parse
    args = parser.parse_args()

    # seed
    utils.set_seed(1234)

    feature_dim, temperature, tau_plus, k = args.feature_dim, args.temperature, args.tau_plus, args.k
    batch_size, epochs, debiased = args.batch_size, args.epochs, args.debiased
    pos_gen_weight = args.pos_gen_weight

    # data prepare
    # Ruoyu - to modify for pos/neg generating
    if args.dataset == 'CIFAR10':
        print('train on CIFAR10')
        train_data = utils.CIFAR10Folder(transform=utils.train_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        memory_data = utils.CIFAR10Folder(transform=utils.test_transform, train=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = utils.CIFAR10Folder(transform=utils.test_transform, train=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif args.dataset == 'CIFAR100':
        print('train on CIFAR100')
        train_data = utils.CIFAR100Folder(transform=utils.train_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        memory_data = utils.CIFAR100Folder(transform=utils.test_transform, train=True)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = utils.CIFAR100Folder(transform=utils.test_transform, train=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif args.dataset == 'STL10':
        print('train on STL10')
        train_data = utils.STL10Folder(transform=utils.train_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        memory_data = utils.STL10Pair(root='data', split='train', transform=utils.test_transform)
        memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        test_data = utils.STL10Folder(transform=utils.test_transform, train=False)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif args.dataset == 'VOC_Dec':
        print('train on VOC_Dec')
        train_data = utils.VOCDecFolder(transform=utils.train_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)
        # memory_data = utils.VOCDecFolder(root='data', split='train', transform=utils.test_transform)
        # memory_loader = DataLoader(memory_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
        # test_data = utils.VOCDecFolder(transform=utils.test_transform, train=False)
        # test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)
    elif args.dataset == 'VOC_Seg':
        print('train on VOC_Seg')
        train_data = utils.VOCSegFolder(transform=utils.train_transform, train=True)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True,
                                  drop_last=True)

    # model setup and optimizer config
    if args.pretrained_backbone:
        print('Using Pretrained ResNet50 as backbone')
        model = PretrainedResNet50(feature_dim).cuda()
        optimizer = optim.Adam(model.g.parameters(), lr=args.lr_enc, weight_decay=1e-6)
        model = nn.DataParallel(model, )
    else:
        print('Train ResNet50 from scratch')
        model = Model(feature_dim).cuda()
        model = nn.DataParallel(model, )
        # pretrain model
        if args.pretrain_model is not None:
            model.load_state_dict(torch.load(args.pretrain_model))
        optimizer = optim.Adam(model.parameters(), lr=args.lr_enc, weight_decay=1e-6)

    # try:
    #     c = len(memory_data.classes)
    # except:
    #     c = len(set(memory_data.targets))
    # print('# Classes: {}'.format(c))


    if args.baseline:
        results = {'train_loss': [],
                   'test_acc@1': [], 'test_acc@5': []}
    else:
        results = {'train_loss_enc': [], 'train_loss_gen': [],
                   'test_acc@1': [], 'test_acc@5': []}

    # training loop
    save_name_pre = '{}_{}_{}_{}_{}_{}'.format(feature_dim, temperature, k, batch_size, epochs,
                                               datetime.now().strftime('%Y%m%d-%H%M'))
    if not os.path.exists('results'):
        os.mkdir('results')
    # if not os.path.exists('results/generated_samples'):
    #     os.mkdir('results/generated_samples')

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        if args.baseline:
            train_loss = train(model, train_loader, optimizer, temperature, debiased, tau_plus)
            # save result
            results['train_loss'].append(train_loss)
        else:
            # traditional saliency method
            # train_loss_enc = train_encoder(model, train_loader, optimizer,
            #                                temperature, pos_gen_weight, neg_gen_weight,
            #                                debiased, tau_plus)

            train_loss_enc = train_encoder(model,
                                           train_loader, optimizer,
                                           temperature, pos_gen_weight,
                                           debiased, tau_plus)


            # save result
            results['train_loss_enc'].append(train_loss_enc)
            results['train_loss_gen'].append(0)

        # test_acc_1, test_acc_5 = test(model, memory_loader, test_loader)
        results['test_acc@1'].append(0)
        results['test_acc@5'].append(0)
        # save statistics
        data_frame = pd.DataFrame(data=results, index=range(1, epoch + 1))
        data_frame.to_csv('results/{}_statistics.csv'.format(save_name_pre), index_label='epoch')

        torch.save(model.state_dict(), 'results/{}_model.pth'.format(save_name_pre))

