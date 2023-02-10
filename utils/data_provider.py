import os
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from model.backbone import *


class HashingDataset(Dataset):
    def __init__(self, data_path, img_filename, label_filename, training=False):
        self.img_path = data_path

        img_filepath = os.path.join(data_path, img_filename)
        fp = open(img_filepath, 'r')
        self.img_filename = [x.strip() for x in fp]
        fp.close()

        label_filepath = os.path.join(data_path, label_filename)
        self.label = np.loadtxt(label_filepath, dtype=np.int64)

        if training:
            self.transform = transforms.Compose([transforms.Resize(256), 
            transforms.RandomCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        else:
            self.transform = transforms.Compose(
                [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()])

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_path, self.img_filename[index]))
        img = img.convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        label = torch.from_numpy(self.label[index]).float()
        return img, label, index

    def __len__(self):
        return len(self.img_filename)


# def load_model(network, bit, path):
#     if network == 'AlexNet':
#         model = AlexNet(bit)
#     elif 'VGG' in network:
#         model = VGG(network, bit)
#     else:
#         model = ResNet(network, bit)
#     model.load_state_dict(torch.load(path))
#     # model = torch.load(path)
#     model = model.cuda()
#     return model

# def save_model(model, path):
#     if not os.path.exists(os.path.dirname(path)):
#         os.makedirs(os.path.dirname(path))
#     torch.save(model.state_dict(), path)


def load_label(data_dir, filename, tensor=False):
    label_filepath = os.path.join(data_dir, filename)
    label = np.loadtxt(label_filepath, dtype=int)
    if tensor:
        label = torch.from_numpy(label).float()
    return label

def generate_code_label(model, data_loader, num_data, bit, num_class):
    B = torch.zeros([num_data, bit]).cuda()
    L = torch.zeros(num_data, num_class).cuda()
    for iter, data in enumerate(data_loader, 0):
        data_input, data_label, data_ind = data
        output = model(data_input.cuda())
        B[data_ind, :] = torch.sign(output.data)
        L[data_ind, :] = data_label.cuda()
    return B, L

def generate_hash_code(model, data_loader, num_data, bit):
    B = np.zeros([num_data, bit], dtype=np.float32)
    for _, data in enumerate(data_loader):
        data_input, _, data_ind = data
        data_input = Variable(data_input.cuda())
        output = model(data_input)
        B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
    return B
