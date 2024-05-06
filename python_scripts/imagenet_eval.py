import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.realpath('..'))

from utils import *

dir_data = '/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val'
dir_uap = '../uaps/imagenet/'
#loader = loader_imgnet(dir_data, 10000, 250) # evaluate on 10,000 validation images
train_loader, test_loader = get_data('imagenet')

targets = [611,734,854,859,497,577,8,5]

# load model

model = torch.load(
    '/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch/models/resnet50_imagenet_finetuned_repaired.pth',
    map_location=torch.device('cpu'))

_, _, _, _, outputs, labels = evaluate(model, test_loader, uap = None)
print('Accuracy:', sum(outputs == labels) / len(labels))


# load pattern
y_target = 611
uap = torch.load(dir_uap + 'uap_%i.pth' % y_target)

# evaluate
_, _, _, _, outputs, labels = evaluate(model, test_loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))
print('Targeted success rate:', sum(outputs == y_target) / len(labels))
