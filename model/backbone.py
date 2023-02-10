import math
import torch
import torch.nn as nn
from torchvision import models


class AlexNet(nn.Module):
    def __init__(self, bit):
        super(AlexNet, self).__init__()
        # pytorch < 0.13
        # original_model = models.alexnet(pretrained=True)
        # pytorch >= 0.13
        original_model = models.alexnet(weights=models.AlexNet_Weights.IMAGENET1K_V1)
        self.features = original_model.features

        cl1 = nn.Linear(256 * 6 * 6, 4096)
        cl1.weight = original_model.classifier[1].weight
        cl1.bias = original_model.classifier[1].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[4].weight
        cl2.bias = original_model.classifier[4].bias

        cl3 = nn.Linear(4096, bit)
        cl3.weight.data.normal_(0, 0.01)
        cl3.bias.data.fill_(0.0)

        self.classifier = nn.Sequential(
            nn.Dropout(),
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            cl3
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x - self.mean) / self.std
        fea = self.features(x)
        fea = fea.view(fea.size(0), -1)
        hash = self.classifier(fea)
        hash = self.tanh(alpha * hash)
        return hash


vgg_dict = {
    "VGG11": models.vgg11,
    "VGG13": models.vgg13,
    "VGG16": models.vgg16,
    "VGG19": models.vgg19,
    "VGG11BN": models.vgg11_bn,
    "VGG13BN": models.vgg13_bn,
    "VGG16BN": models.vgg16_bn,
    "VGG19BN": models.vgg19_bn
}


class VGG(nn.Module):
    def __init__(self, model_name, bit):
        super(VGG, self).__init__()
        # pytorch < 0.13
        original_model = vgg_dict[model_name](pretrained=True)
        # pytorch >= 0.13
        # original_model = vgg_dict[model_name](weights=models.VGG16_Weights.IMAGENET1K_V1)
        self.features = original_model.features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        cl1 = nn.Linear(25088, 4096)
        cl1.weight = original_model.classifier[0].weight
        cl1.bias = original_model.classifier[0].bias

        cl2 = nn.Linear(4096, 4096)
        cl2.weight = original_model.classifier[3].weight
        cl2.bias = original_model.classifier[3].bias

        cl3 = nn.Linear(4096, bit)
        cl3.weight.data.normal_(0, 0.01)
        cl3.bias.data.fill_(0.0)

        self.classifier = nn.Sequential(
            cl1,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl2,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            cl3
        )

        self.tanh = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x - self.mean) / self.std
        fea = self.features(x)
        fea = torch.flatten(fea, 1)
        hash = self.classifier(fea)
        hash = self.tanh(alpha * hash)
        return hash


resnet_dict = {
    "ResNet18": models.resnet18,
    "ResNet34": models.resnet34,
    "ResNet50": models.resnet50,
    "ResNet101": models.resnet101,
    "ResNet152": models.resnet152
}


class ResNet(nn.Module):
    def __init__(self, model_name, hash_bit):
        super(ResNet, self).__init__()
        model_resnet = resnet_dict[model_name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                             self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)

        self.activation = nn.Tanh()
        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x, alpha=1):
        x = (x-self.mean)/self.std
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        hash = self.activation(alpha*y)
        return hash


densenet_dict = {'DenseNet161': models.densenet161}

class DenseNetFc(nn.Module):
  def __init__(self, name, hash_bit):
    super(DenseNetFc, self).__init__()
    model_dense = densenet_dict[name](pretrained=True)
    self.features = model_dense.features
    #self.classifier = nn.Sequential()
    self.avg_pool = nn.AvgPool2d(7)
    self.classifier = model_dense.classifier
    self.feature_layers = nn.Sequential(self.features, self.avg_pool) #nn.Sequential(self.features, self.classifier)

    self.hash_layer = nn.Linear(model_dense.classifier.in_features, hash_bit)
    self.hash_layer.weight.data.normal_(0, 0.01)
    self.hash_layer.bias.data.fill_(0.0)
    self.iter_num = 0
    self.__in_features = hash_bit
    self.step_size = 200
    self.gamma = 0.005
    self.power = 0.5
    self.init_scale = 1.0
    self.activation = nn.Tanh()
    self.scale = self.init_scale

    self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
    self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

  def forward(self, x):
    x = (x - self.mean) / self.std
    if self.training:
        self.iter_num += 1
    x = self.feature_layers(x)
    x = x.view(x.size(0), -1)
    y = self.hash_layer(x)
    if self.iter_num % self.step_size==0:
        self.scale = self.init_scale * (math.pow((1.+self.gamma*self.iter_num), self.power))
    y = self.activation(self.scale*y)
    return y

  def output_num(self):
    return self.__in_features



class AlexNetFc(nn.Module):
    def __init__(self, hash_bit):
        super(AlexNetFc, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_alexnet.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        # self.use_hashnet = use_hashnet
        self.hash_layer = nn.Linear(model_alexnet.classifier[6].in_features,
                                    hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow(
                (1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


class VGGFc(nn.Module):
    def __init__(self, name, hash_bit):
        super(VGGFc, self).__init__()
        model_vgg = vgg_dict[name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module("classifier" + str(i),
                                       model_vgg.classifier[i])
        self.feature_layers = nn.Sequential(self.features, self.classifier)

        # self.use_hashnet = use_hashnet
        self.hash_layer = nn.Linear(model_vgg.classifier[6].in_features,
                                    hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

        self.mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]]).cuda()
        self.std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]]).cuda()

    def forward(self, x):
        x = (x - self.mean) / self.std
        if self.training:
            self.iter_num += 1
        x = self.features(x)
        x = x.view(x.size(0), 25088)
        x = self.classifier(x)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow(
                (1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features


class ResNetFc(nn.Module):
    def __init__(self, name, hash_bit):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[name](pretrained=True)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.feature_layers = nn.Sequential(self.conv1, self.bn1, self.relu, self.maxpool, \
                             self.layer1, self.layer2, self.layer3, self.layer4, self.avgpool)

        self.hash_layer = nn.Linear(model_resnet.fc.in_features, hash_bit)
        self.hash_layer.weight.data.normal_(0, 0.01)
        self.hash_layer.bias.data.fill_(0.0)
        self.iter_num = 0
        self.__in_features = hash_bit
        self.step_size = 200
        self.gamma = 0.005
        self.power = 0.5
        self.init_scale = 1.0
        self.activation = nn.Tanh()
        self.scale = self.init_scale

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        y = self.hash_layer(x)
        if self.iter_num % self.step_size == 0:
            self.scale = self.init_scale * (math.pow(
                (1. + self.gamma * self.iter_num), self.power))
        y = self.activation(self.scale * y)
        return y

    def output_num(self):
        return self.__in_features
