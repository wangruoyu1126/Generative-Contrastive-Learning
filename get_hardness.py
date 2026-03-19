# Prepare data
from torch.utils.data import DataLoader
import utils
from tqdm import tqdm
import torch
from saliency import *

batch_size = 1

utils.set_seed(1234)


test_data = utils.CIFAR10(root='data', train=False, transform=utils.test_transform, download=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=16, pin_memory=True)

# Load model
# baseline_resnet = 'results/baseline_20220709/128_0.5_200_256_1000_20220709-1641_model.pth'
baseline_linear = 'results/baseline_20220709/linear_model.pth'

from model import Model
import torch.nn as nn

class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()

        # encoder
        model = Model().cuda()
        model = nn.DataParallel(model)
        # model.load_state_dict(torch.load(pretrained_path))

        self.f = model.module.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


net = Net(num_class=len(test_data.classes)).cuda()
net.load_state_dict(torch.load(baseline_linear))
net.eval()

total_loss, total_correct_1, total_correct_5, total_num, data_bar = 0.0, 0.0, 0.0, 0, tqdm(test_loader)
loss_criterion = nn.CrossEntropyLoss()

# Make Prediction
with torch.no_grad():
    for data, target in data_bar:
        data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
        out = net(data)
        print(out)

        # loss = loss_criterion(out, target)
        # total_num += data.size(0)
        # total_loss += loss.item() * data.size(0)

        prediction = torch.argsort(out, dim=-1, descending=True)
        print(prediction)
        # total_correct_1 += torch.sum((prediction[:, 0:1] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        # total_correct_5 += torch.sum((prediction[:, 0:5] == target.unsqueeze(dim=-1)).any(dim=-1).float()).item()
        #
        # data_bar.set_description('{} Epoch: Loss: {:.4f} ACC@1: {:.2f}% ACC@5: {:.2f}%'
        #                          .format('Test', total_loss / total_num,
        #                                  total_correct_1 / total_num * 100, total_correct_5 / total_num * 100))




# Cauculate Hardness




















