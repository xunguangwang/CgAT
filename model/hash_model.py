import os
import torch
import torch.optim as optim

from model.backbone import *
from utils.hamming_matching import *
from .hashing_loss import *


class HashingModel(object):
    def __init__(self, args):
        super(HashingModel, self).__init__()
        class_dic = {'CIFAR-10': 10, 'ImageNet': 100, 'FLICKR-25K': 38, 'NUS-WIDE':21, 'MS-COCO': 80}
        self.num_classes = class_dic[args.dataset]
        self.bit = args.bit
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.backbone = args.backbone
        self.method = args.method
        self.model_name = 'Hash_{}_{}_{}_{}'.format(args.dataset, self.method, self.backbone, args.bit)
        print(self.model_name)
        self.args = args

        self._build_graph()

    def _build_graph(self):
        if self.backbone == 'AlexNet':
            self.model = AlexNet(self.args.bit)
        elif 'VGG' in self.backbone:
            self.model = VGG(self.backbone, self.args.bit)
        elif 'ResNet' in self.backbone:
            self.model = ResNet(self.backbone, self.args.bit)
        else:
            self.model = DenseNetFc(self.backbone, self.args.bit)
        self.model = self.model.cuda()

    def load_model(self):
        self.model.load_state_dict(torch.load(
            os.path.join(self.args.save, self.model_name + '.pth')))
        self.model = self.model.cuda()

    def save_model(self):
        if not os.path.exists(self.args.save):
            os.makedirs(self.args.save)
        
        torch.save(self.model.state_dict(), os.path.join(self.args.save, self.model_name + '.pth'))

    def get_hashing_loss(self, num_train):
        if self.method == 'DPSH':
            return DPSHLoss(num_train, self.bit, self.num_classes)
        elif self.method == 'HashNet':
            return HashNetLoss(num_train, self.bit, self.num_classes)
        elif self.method == 'DSDH':
            return DSDHLoss(num_train, self.bit, self.num_classes)
        elif self.method == 'DSDH-C':
            return DSDHCLoss(num_train, self.bit, self.num_classes)
        elif self.method == 'DCH':
            return DCHLoss(self.batch_size, self.bit)
        elif self.method == 'GreedyHash':
            return GreedyHashLoss(self.bit, self.num_classes)
        elif self.method == 'CSQ':
            is_single_label = self.args.dataset not in {'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'}
            return CSQLoss(self.bit, self.num_classes, is_single_label)
        elif self.method == 'CSQ-C':
            is_single_label = self.args.dataset not in {'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'}
            return CSQCLoss(self.bit, self.num_classes, is_single_label)
        else:
            return DPHLoss(num_train, self.bit, self.num_classes)
    
    def adjust_learning_rate(self, optimizer, epoch):
        lr = self.lr * (0.1**(epoch // (self.args.n_epochs // 3)))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        return optimizer

    def generate_code(self, data_loader, num_data):
        B = np.zeros([num_data, self.bit], dtype=np.float32)
        for iter, data in enumerate(data_loader, 0):
            data_input, _, data_ind = data
            data_input = data_input.cuda()
            output = self.model(data_input)
            B[data_ind.numpy(), :] = torch.sign(output.cpu().data).numpy()
            # print('{}:{}'.format(iter, torch.sign(output.cpu().data).numpy()))
        return B

    def train(self, train_loader, num_train):
        self.model.train()
        criterion = self.get_hashing_loss(num_train)
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.args.lr, momentum=self.args.momentum,
                              weight_decay=self.args.wd)

        for epoch in range(self.args.n_epochs):
            if 'DSDH' in self.method:
                criterion.update_band_W()
            epoch_loss = 0.0
            # training epoch
            for iter, data in enumerate(train_loader):
                input, label, ind = data
                label = torch.squeeze(label)
                input, label = input.cuda(), label.cuda()

                self.model.zero_grad()
                train_outputs = self.model(input)
                loss = criterion(train_outputs, label, ind)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print('Epoch: %3d/%3d\tTrain_loss: %3.5f' %
                  (epoch + 1, self.args.n_epochs,
                   epoch_loss / len(train_loader)))
            optimizer = self.adjust_learning_rate(optimizer, epoch)

        self.save_model()

    def test(self, database_loader, test_loader, database_labels, test_labels,
             num_database, num_test):
        self.model.eval()
        qB = self.generate_code(test_loader, num_test)
        dB = self.generate_code(database_loader, num_database)
        map = CalcTopMap(qB, dB, test_labels, database_labels, 5000)
        print('[retrieval database] Tested Top MAP: %3.5f' % (map))
        map = CalcMap(qB, dB, test_labels, database_labels)
        print('[retrieval database] Tested MAP: %3.5f' % (map))
