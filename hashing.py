import os
os.environ["CUDA_VISIBLE_DEVICES"]='0'
import argparse
from torch.utils.data import DataLoader

from model.hash_model import *
from utils.data_provider import *


parser = argparse.ArgumentParser()
# description of data
parser.add_argument('--dataset_name', dest='dataset', default='NUS-WIDE', choices=['CIFAR-10', 'ImageNet', 'FLICKR-25K', 'NUS-WIDE', 'MS-COCO'], help='name of the dataset')
parser.add_argument('--data_dir', dest='data_dir', default='../data/', help='path of the dataset')
parser.add_argument('--database_file', dest='database_file', default='database_img.txt', help='the image list of database images')
parser.add_argument('--train_file', dest='train_file', default='train_img.txt', help='the image list of training images')
parser.add_argument('--test_file', dest='test_file', default='test_img.txt', help='the image list of test images')
parser.add_argument('--database_label', dest='database_label', default='database_label.txt', help='the label list of database images')
parser.add_argument('--train_label', dest='train_label', default='train_label.txt', help='the label list of training images')
parser.add_argument('--test_label', dest='test_label', default='test_label.txt', help='the label list of test images')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=64, help='number of images in one batch')
# model
parser.add_argument('--hashing_method', dest='method', default='DSDH-C', choices=['DPH', 'DPSH', 'HashNet','DSDH', 'DCH', 'GreedyHash', 'CSQ', 'CSQ-C'], help='deep hashing methods')
parser.add_argument('--backbone', dest='backbone', default='VGG11', choices= ['AlexNet', 'VGG11', 'VGG16', 'VGG19', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'DenseNet161'], help='backbone network')
parser.add_argument('--code_length', dest='bit', type=int, default=32, help='length of the hashing code')
# training or test
parser.add_argument('--train', dest='train', type=bool, default=True, choices=[True, False], help='to train or not')
parser.add_argument('--test', dest='test', type=bool, default=True, choices=[True, False], help='to test or not')
parser.add_argument('--load_model', dest='load', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
parser.add_argument('--checkpoint_dir', dest='save', default='checkpoint/', help='models are saved here')
parser.add_argument('--n_epochs', dest='n_epochs', type=int, default=100, help='number of epoch')
parser.add_argument('--learning_rate', dest='lr', type=float, default=0.01, help='initial learning rate for sgd')
parser.add_argument('--momentum', dest='momentum', type=float, default=0.9, help='momentum for SGD')
parser.add_argument('--weight_decay', dest='wd', type=float, default=5e-4, help='weight decay for SGD')
args = parser.parse_args()


data_dir = os.path.join(args.data_dir, args.dataset)
dset_database = HashingDataset(data_dir, args.database_file, args.database_label)
dset_train = HashingDataset(data_dir, args.train_file, args.train_label, training=True)
dset_test = HashingDataset(data_dir, args.test_file, args.test_label)
num_database, num_test, num_train = len(dset_database), len(dset_test), len(dset_train)
database_loader = DataLoader(dset_database, batch_size=args.batch_size, shuffle=False, num_workers=4)
train_loader = DataLoader(dset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dset_test, batch_size=args.batch_size, shuffle=False, num_workers=4)

database_labels = load_label(data_dir, args.database_label)
test_labels = load_label(data_dir, args.test_label)

model = HashingModel(args)

if args.train:
    model.train(train_loader, num_train)

if args.test:
    model.load_model()
    model.test(database_loader, test_loader, database_labels, test_labels, num_database, num_test)
