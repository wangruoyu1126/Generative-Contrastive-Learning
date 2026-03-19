# import utils
# from model import PretrainedResNet50, Model
# from torchvision.models.resnet import resnet50
# from torchsummary import summary
#
#
# model = PretrainedResNet50().cuda()
# # print(model)
# print(summary(model, (3,224,224)))





import os
import torch
from saliency import *

img_path = 'test.jpg'
org_img = cv2.imread(img_path)
org_img = np.expand_dims(org_img, axis=0)

sal_org_img = get_bin_saliency_batch(org_img)
sal_org_img = torch.from_numpy(np.stack(sal_org_img, axis=0))

mask_org_img = sal_org_img.unsqueeze(dim=3).expand(-1, -1, -1, 3)

pos_gen = org_img.copy()
neg_gen = org_img.copy()

pos_gen[mask_org_img == False] = 0
neg_gen[mask_org_img] = 0

cv2.imwrite('test_pos.jpg', pos_gen[0].astype('uint8'))
cv2.imwrite('test_mask.jpg', 255 * mask_org_img[0].numpy().astype('uint8'))
cv2.imwrite('test_neg.jpg', neg_gen[0].astype('uint8'))








