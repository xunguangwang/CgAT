import os
from model.backbone import *


def load_model(network, bit, path):
    if network == 'AlexNet':
        model = AlexNet(bit)
    elif 'VGG' in network:
        model = VGG(network, bit)
    elif 'ResNet' in network:
        model = ResNet(network, bit)
    else:
        model = DenseNetFc(network, bit)
    model.load_state_dict(torch.load(path))
    # model = torch.load(path)
    model = model.cuda()
    return model

def save_model(model, path):
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    torch.save(model.state_dict(), path)
