import torch
import torch.nn as nn
from model import Model
import cv2
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image

class Net(nn.Module):
    def __init__(self, num_class):
        super(Net, self).__init__()

        # encoder
        model = Model().cuda()
        model = nn.DataParallel(model)

        self.f = model.module.f
        # classifier
        self.fc = nn.Linear(2048, num_class, bias=True)

    def forward(self, x):
        x = self.f(x)
        feature = torch.flatten(x, start_dim=1)
        out = self.fc(feature)
        return out


model_folder = 'results/CIFAR10/baseline_3views_test_pos1'
model_path = model_folder + '/linear_model.pth'

model = Net(num_class=10).cuda()
model.load_state_dict(torch.load(model_path))
model.eval()

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])



test_data_path = 'data/CIFAR10_Label/test/img'
pred_list = []

for i in range(10000):
    # print('img number', i)
    img_name = test_data_path + '/{}.jpg'.format(i)
    img = cv2.imread(img_name)
    img = Image.fromarray(img)
    img = test_transform(img)
    img = torch.unsqueeze(img, 0)
    img = img.cuda(non_blocking=True)


    out = model(img)
    prediction = torch.argsort(out, dim=-1, descending=True)
    top1_pred = prediction[0][0].item()
    pred_list.append(top1_pred)


with open(model_folder + '/test_pred.txt', 'w') as f:
    for line in pred_list:
        f.write(f"{line}\n")


































