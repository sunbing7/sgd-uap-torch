import matplotlib.pyplot as plt
import numpy as np
import os
import sys
sys.path.append(os.path.realpath('..'))

import torch
from utils import *

import warnings
warnings.filterwarnings("ignore")

ADAPTIVE_ATTACK = True


targets = [611, 734, 854, 859, 497, 577, 8, 5]


from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate

dir_data = '/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val'
dir_uap = '../uaps/imagenet/'

#test_loader = loader_imgnet(dir_data, 2000, 100) # adjust batch size as appropriate
#train_loader = test_loader
train_loader, test_loader = get_data('imagenet')

# load model
if ADAPTIVE_ATTACK:
    model = model_imgnet('resnet50', adaptive=True)
else:
    model = model_imgnet('resnet50')

# clean accuracy
#_, _, _, _, outputs, labels = evaluate(model, test_loader)
#print('Accuracy:', sum(outputs == labels) / len(labels))


nb_epoch = 2
eps = 10 / 255
beta = 12
step_decay = 0.6

for y_target in targets:
    print('Target class {}'.format(y_target))
    uap, losses = uap_sgd(model, train_loader, nb_epoch, eps, beta, step_decay, y_target=y_target)

    # evaluate
    _, _, _, _, outputs, labels = evaluate(model, test_loader, uap = uap)
    print('Accuracy:', sum(outputs == labels) / len(labels))
    print('Targeted success rate:', sum(outputs == y_target) / len(labels))

    if ADAPTIVE_ATTACK:
        torch.save(uap, '../uaps/imagenet/uap_' + str(y_target) + 'adaptive.pth')
    else:
        torch.save(uap, '../uaps/imagenet/uap_' + str(y_target) + '.pth')