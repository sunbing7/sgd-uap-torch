import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch

sys.path.append(os.path.realpath('..'))

from utils import loader_imgnet, model_imgnet, evaluate

dir_data = '/root/autodl-tmp/sunbing/workspace/uap/data/imagenet/validation_folder/val'
dir_uap = '../uaps/imagenet/'
loader = loader_imgnet(dir_data, 10000, 250) # evaluate on 10,000 validation images

# load model
model = model_imgnet('resnet50')

_, _, _, _, outputs, labels = evaluate(model, loader, uap = None)
print('Accuracy:', sum(outputs == labels) / len(labels))

# load pattern
uap = torch.load(dir_uap + 'sgd-resnet50-eps10.pth')

# visualize
uap_max = torch.max(uap)
plt.imshow(np.transpose(((uap / uap_max) + 1) / 2, (1, 2, 0)))

# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))

# load pattern
y_target = 611
uap = torch.load(dir_uap + 'sgd-tgt%i-resnet50-eps10.pth' % y_target)

# visualize
uap_max = torch.max(uap)
plt.imshow(np.transpose(((uap / uap_max) + 1) / 2, (1, 2, 0)))

# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))
print('Targeted success rate:', sum(outputs == y_target) / len(labels))

'''
# load pattern
uap = torch.load(dir_uap + 'sgd-resnet50_SIN-eps10.pth')

# visualize
uap_max = torch.max(uap)
plt.imshow(np.transpose(((uap / uap_max) + 1) / 2, (1, 2, 0)))

# evaluate
_, _, _, _, outputs, labels = evaluate(model, loader, uap = uap)
print('Accuracy:', sum(outputs == labels) / len(labels))
'''