import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.realpath('..'))

from attacks import uap_sgd
from utils import loader_imgnet, model_imgnet, evaluate

dir_data = '/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val'
dir_uap = '../uaps/imagenet/'

loader = loader_imgnet(dir_data, 2000, 100) # adjust batch size as appropriate

# load model
model = model_imgnet('resnet50')

# clean accuracy
_, _, _, _, outputs, labels = evaluate(model, loader)
print('Accuracy:', sum(outputs == labels) / len(labels))


nb_epoch = 10
eps = 10 / 255
y_target = 815
beta = 12
step_decay = 0.6
uap, losses = uap_sgd(model, loader, nb_epoch, eps, beta, step_decay, y_target=y_target)

# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))
print('Targeted success rate:', sum(outputs == y_target) / len(labels))