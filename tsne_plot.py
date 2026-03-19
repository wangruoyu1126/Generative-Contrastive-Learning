from tsne_torch import TorchTSNE as TSNE
import torch
import seaborn as sns
import matplotlib.pyplot as plt

import utils
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from model import Model

batch_size = 8

# test_data = utils.CIFAR10Folder(transform=utils.test_transform, train=False)
# test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
#                          pin_memory=True)

test_data = utils.STL10Folder(transform=utils.test_transform, train=False)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16,
                         pin_memory=True)



# model_path = 'results/CIFAR10/SimCLR/128_0.5_200_256_200_20221104-1010_model.pth'
# model_path = 'results/CIFAR10/SimCLR+ours/128_0.5_200_256_200_20221103-0900_model.pth'
# model_path = 'results/CIFAR10/DCL/128_0.5_200_256_200_20221105-1304_model.pth'
# model_path = 'results/CIFAR10/DCL+ours/128_0.5_200_256_200_20221104-1431_model.pth'
# model_path = 'results/CIFAR10/HCL/128_0.5_200_256_200_20221025-1027_model.pth'
# model_path = 'results/CIFAR10/HCL+ours/128_0.5_200_256_200_20221025-1023_model.pth'



# model_path = 'results/STL10/SimCLR/128_0.5_200_256_200_20221031-1609_model.pth'
# model_path = 'results/STL10/SimCLR+ours/128_0.5_200_256_200_20221030-1012_model.pth'
# model_path = 'results/STL10/DCL/128_0.5_200_256_200_20221029-1308_model.pth'
# model_path = 'results/STL10/DCL+ours/128_0.5_200_256_200_20221022-2202_model.pth'
# model_path = 'results/STL10/HCL/128_0.5_200_256_200_20221027-1134_model.pth'
model_path = 'results/STL10/HCL+ours/128_0.5_200_256_200_20221027-1910_model.pth'



import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, pretrained_path):
        super(Net, self).__init__()

        # encoder
        model = Model().cuda()
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(pretrained_path))

        self.f = model.module.f
        self.g = model.module.g
        # classifier
        # self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.g(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


model = Net(pretrained_path=model_path).cuda()
for param in model.f.parameters():
    param.requires_grad = False
for param in model.g.parameters():
    param.requires_grad = False










feature_bank = {'feature':[], 'target':[]}

batch_num = 0
for data, _, _, _, target in tqdm(test_loader):
    data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
    feature, out = model(data)
    # print('feature shape is', feature.shape)
    # print('target shape is', target.shape)
    feature_bank['feature'].append(out)
    feature_bank['target'].append(target)
    batch_num += 1
    if batch_num >= 3000/batch_size:
        break

all_features = torch.cat(feature_bank['feature'])
all_targets = torch.cat(feature_bank['target']).tolist()



# all_features = torch.rand(500, 100)
# all_targets = torch.randint(low=0, high=10, size=(1,500)).tolist()[0]
# print(all_targets)

# feature_tsne = TSNE(n_components=2, perplexity=30, n_iter=10000, verbose=True, initial_dims=500).fit_transform(all_features)


all_features = all_features.cpu().detach().numpy()
print('all_features shape is', all_features.shape)
print('all_targets shape is', len(all_targets))

import numpy as np
from sklearn.manifold import TSNE

# all_features = PCA(n_components=50).fit_transform(all_features)
feature_tsne = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(all_features)

# print('all_features_pca shape is', all_features_pca.shape)
# print('feature_tsne shape is', feature_tsne.shape)

target_dict = {0: ['airplane', 'black'],
               1: ['automobile', 'red'],
               2: ['bird', 'orange'],
               3: ['cat', 'blue'],
               4: ['deer', 'purple'],
               5: ['dog', 'silver'],
               6: ['frog', 'lime'],
               7: ['horse', 'aqua'],
               8: ['ship', 'green'],
               9: ['truck', 'yellow']}


# all_target_label = [target_dict[item] for item in all_targets]

color_list = [target_dict[lbl][1] for lbl in all_targets]

# for label_idx in range(10):
#     plt.scatter(
#         feature_tsne[all_targets == label_idx, 0],
#         feature_tsne[all_targets == label_idx, 1],
#         c=target_dict[label_idx][1],
#         s=5)


plt.scatter(feature_tsne[:, 0], feature_tsne[:, 1], c=color_list, s=5)
plt.axis('off')

plt.show()
plt.savefig('tsne_out.png')
plt.close()








