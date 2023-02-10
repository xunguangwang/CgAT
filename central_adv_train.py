import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from utils.data_provider import *
from utils.model_loader import *
from utils.hamming_matching import *
from model.backbone import *
from model.hashing_loss import *


def adv_loss(noisy_output, target_hash):
    loss = torch.mean(noisy_output * target_hash)
    return loss

def hash_adv(model, query, target_hash, epsilon, step=2, iteration=7, randomize=True):
    delta = torch.zeros_like(query).cuda()
    if randomize:
        delta.uniform_(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
    delta.requires_grad = True

    for i in range(iteration):
        noisy_output = model(query + delta)
        loss = adv_loss(noisy_output, target_hash.detach())
        loss.backward()

        delta.data = delta - step/255 * torch.sign(delta.grad.detach())
        delta.data = delta.data.clamp(-epsilon, epsilon)
        delta.data = (query.data + delta.data).clamp(0, 1) - query.data
        delta.grad.zero_()

    return query + delta.detach()

def hash_center_code(y, B, L, bit):
    code = torch.zeros(y.size(0), bit).cuda()
    for i in range(y.size(0)):
        l = y[i].repeat(L.size(0), 1)
        w = torch.sum(l*L, dim=1) / torch.sum(torch.sign(l + L), dim=1)
        w1 = w.repeat(bit, 1).t()
        w2 = 1 - torch.sign(w1)
        c = w2.sum()/bit
        w1 = 1 - w2
        code[i] = torch.sign(torch.sum(c*w1*B-(L.size(0)-c)*w2*B, dim=0))
    return code


classes_dic = {'FLICKR-25K': 38, 'NUS-WIDE':21, 'MS-COCO': 80, 'ImageNet': 100, 'CIFAR-10': 10}
dataset = 'NUS-WIDE'
DATA_DIR = '../data/{}'.format(dataset)
DATABASE_FILE = 'database_img.txt'
TRAIN_FILE = 'train_img.txt'
TEST_FILE = 'test_img.txt'
DATABASE_LABEL = 'database_label.txt'
TRAIN_LABEL = 'train_label.txt'
TEST_LABEL = 'test_label.txt'
num_classes = classes_dic[dataset]
model_name = 'DPSH'
backbone = 'VGG11'
batch_size = 32
bit = 32
epsilon = 8 / 255.0
epochs = 30
iteration = 7

lr = 1e-2
momentum=0.9
weight_decay = 5e-4

dset_database = HashingDataset(DATA_DIR, DATABASE_FILE, DATABASE_LABEL)
dset_train = HashingDataset(DATA_DIR, TRAIN_FILE, TRAIN_LABEL, training=True)
dset_test = HashingDataset(DATA_DIR, TEST_FILE, TEST_LABEL)
database_loader = DataLoader(dset_database, batch_size=batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=batch_size, shuffle=False, num_workers=4)
num_database, num_train, num_test = len(dset_database), len(dset_train), len(dset_test)

database_labels = load_label(DATA_DIR, DATABASE_LABEL, True)
train_labels = load_label(DATA_DIR, TRAIN_LABEL, True).cuda()
test_labels = load_label(DATA_DIR, TEST_LABEL, True)
target_labels = database_labels.unique(dim=0)


model_path = 'checkpoint/Hash_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
robust_model_path = 'checkpoint/DHCAT_{}_{}_{}_{}.pth'.format(dataset, model_name, backbone, bit)
print(robust_model_path)
model = load_model(backbone, bit, model_path)


opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
lr_steps = epochs * len(train_loader)
scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=[lr_steps / 2, lr_steps * 3 / 4], gamma=0.1)
train_B, train_L = generate_code_label(model, train_loader, num_train, bit, num_classes)
U_ben = torch.zeros(num_train, bit).cuda()
U_ben.data = train_B.data
U_adv = torch.zeros(num_train, bit).cuda()
B_ben = torch.zeros(num_train, bit).cuda()
B_ben.data = train_B.data
hash_criterion = get_hashing_loss(model_name, num_train, bit, num_classes, batch_size)
hash_criterion.preinstall(U_ben, train_L)
# adversarial training
for epoch in range(epochs):
    if 'DSDH' in model_name:
        hash_criterion.update_band_W()
    epoch_loss = 0
    for it, data in enumerate(train_loader):
        x, y, index = data
        x = x.cuda()
        y = y.cuda()
        batch_size_ = index.size(0)

        center_codes_curr = hash_center_code(y, B_ben, train_L, bit)
        x_adv = hash_adv(model, x, center_codes_curr, epsilon, step=2, iteration=iteration, randomize=True)
        x_adv = x_adv.detach()

        model.zero_grad()
        output_adv = model(x_adv)
        output_ben = model(x)
        B_ben[index, :] = torch.sign(output_ben.data)
        U_ben[index, :] = output_ben.data

        loss_hash_ben = hash_criterion(output_ben, y, index)

        center_codes = hash_center_code(y, B_ben, train_L, bit)
        loss_hash_adv = -torch.mean((output_adv * center_codes))
        loss = loss_hash_ben + loss_hash_adv
        loss.backward()
        opt.step()
        scheduler.step()
        epoch_loss += loss.item()

        if it % 50 == 0:
            print('epoch: {:2d}, step: {:3d}, lr: {:.5f}, loss: {:.5f}, loss_ben: {:.5f}, loss_adv: {:.5f}'.format(
                epoch, it, scheduler.get_last_lr()[0], loss, loss_hash_ben, loss_hash_adv))
    
    print('Epoch: %3d/%3d\tTrain_loss: %3.5f \n' % (epoch, epochs, epoch_loss / len(train_loader)))


save_model(model, robust_model_path)
