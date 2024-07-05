'''
Functions for:
- Loading models, datasets
- Evaluating on datasets with or without UAP
'''

import multiprocessing
import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torchvision

from googlenet_imp import googlenet as my_googlenet
from shufflenetv2 import shufflenetv2
from mobilenet import MobileNet

from models_cifar import *
from torch.utils import model_zoo
from torch.utils.data import DataLoader, Subset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from data_sat import *


# Copy and edit this file to config/config.py
MY_PATH = "/root/autodl-tmp/sunbing/workspace/uap"

PROJECT_PATH = MY_PATH + "/uap_virtual_data.pytorch"                    # Directory to this project
DATASET_BASE_PATH = MY_PATH + "/data"                                                   # Directory path wehre pytorch datasets are stored
IMAGENET_PATH = MY_PATH + "/data/imagenet"                                              # Directory to ImageNet for Pytorch
CALTECH_PATH = MY_PATH + "/data/caltech"                                                # Directory to caltech for Pytorch
ASL_PATH = MY_PATH + "/data/asl"                                                            # Directory to asl for Pytorch

CIFAR_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR_STD = [0.2023, 0.1994, 0.2010]

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]


class Normalizer(nn.Module):
    def __init__(self, mean, std):
        super(Normalizer, self).__init__()
        if not isinstance(mean, torch.Tensor):
            mean = torch.tensor(mean)
        if not isinstance(std, torch.Tensor):
            std = torch.tensor(std)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)
        
    def forward(self, tensor):
        return normalize_fn(tensor, self.mean, self.std)
    
    def extra_repr(self):
        return 'mean={}, std={}'.format(self.mean, self.std)
    

def normalize_fn(tensor, mean, std):
    """
    Differentiable version of torchvision.functional.normalize
    - default assumes color channel is at dim = 1
    """
    mean = mean[None, :, None, None]
    std = std[None, :, None, None]
    return tensor.sub(mean).div(std)


'''
Load pre-trained ImageNet models

For models pre-trained on Stylized-ImageNet:
[ICLR 2019] ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness
Paper: https://openreview.net/forum?id=Bygh9j09KX
Code: https://github.com/rgeirhos/texture-vs-shape
'''    
def model_imgnet(model_name, adaptive=False):
    '''
    model_name options:
    resnet50_SIN       trained on Stylized only
    resnet50_SIN-IN    trained on ImageNet + Stylized
    resnet50_SIN-2IN   trained on ImageNet + Stylized, then fine-tuned on ImageNet
    
    or load torchvision.models pre-trained on ImageNet: https://pytorch.org/docs/stable/torchvision/models.html
    '''
    
    if model_name[:12] == 'resnet50_SIN':
        model_urls = {
            'resnet50_SIN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/6f41d2e86fc60566f78de64ecff35cc61eb6436f/resnet50_train_60_epochs-c8e5653e.pth.tar',
            'resnet50_SIN-IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_train_45_epochs_combined_IN_SF-2a0d100e.pth.tar',
            'resnet50_SIN-2IN': 'https://bitbucket.org/robert_geirhos/texture-vs-shape-pretrained-models/raw/60b770e128fffcbd8562a3ab3546c1a735432d03/resnet50_finetune_60_epochs_lr_decay_after_30_start_resnet50_train_45_epochs_combined_IN_SF-ca06340c.pth.tar',
        }
        model = torchvision.models.resnet50(pretrained=False)
        model = nn.DataParallel(model).cuda()
        checkpoint = model_zoo.load_url(model_urls[model_name])
        model.load_state_dict(checkpoint['state_dict'])
        model = nn.DataParallel(model).cuda()
    
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean = IMGNET_MEAN, std = IMGNET_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    print("Model loading complete.")
    
    return model


def load_model(args, model_path, mean, std):
    if args.adaptive_attack:
        model = torch.load(
            model_path,
            map_location=torch.device('cpu'))
        model.eval()
    else:
        model = eval("torchvision.models.{}(pretrained=True)".format(args.model_name))

    model = nn.DataParallel(model).cuda()

    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean=mean, std=std)
    model = nn.Sequential(normalize, model)
    model = model.cuda()
    print("Model loading complete.")

    return model


def get_network(model_arch, num_classes=1000):

    if model_arch == "alexnet":
        net = models.alexnet(pretrained=True)
    elif model_arch == "googlenet":
        net = my_googlenet(pretrained=True)
    elif model_arch == "vgg16":
        net = models.vgg16(pretrained=True)
    elif model_arch == "vgg19":
        net = models.vgg19(pretrained=True)
    elif model_arch == "resnet18":
        net = models.resnet18(pretrained=True)
    elif model_arch == "resnet34":
        net = models.resnet34(pretrained=True)
    elif model_arch == "resnet50":
        net = models.resnet50(pretrained=True)
    elif model_arch == "resnet101":
        net = models.resnet101(pretrained=True)
    elif model_arch == "resnet152":
        net = models.resnet152(pretrained=True)
    elif model_arch == "inception_v3":
        net = models.inception_v3(pretrained=True)
    elif model_arch == 'shufflenetv2':
        net = shufflenetv2(num_classes=num_classes,pretrained=False)
    elif model_arch == 'mobilenet':
        net = MobileNet(num_classes=num_classes, pretrained=False)
    else:
        raise ValueError("Network {} not supported".format(model_arch))
    return net


# Load pre-trained CIFAR-10 models
def model_cifar(model_name, ckpt_path):
    '''
    CIFAR-10 model implementations from:
    https://github.com/kuangliu/pytorch-cifar
    '''
    if model_name == 'resnet18':
        model = ResNet18()
    elif model_name == 'vgg16':
        model = VGG('VGG16')
        
    model = model.cuda()
    model = torch.nn.DataParallel(model)
    
    # Load saved weights and stats
    checkpoint = torch.load(ckpt_path)
    model.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']
    
    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean = CIFAR_MEAN, std = CIFAR_STD)
    model = nn.Sequential(normalize, model)
    model = model.cuda()

    return model, best_acc


# dataloader for ImageNet
def loader_imgnet(dir_data, nb_images = 50000, batch_size = 100, img_size = 224):
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor()
    ])
    
    val_dataset = ImageFolder(dir_data, val_transform)
    
    # Random subset if not using the full 50,000 validation set
    if nb_images < 50000:
        np.random.seed(0)
        sample_indices = np.random.permutation(range(50000))[:nb_images]
        val_dataset = Subset(val_dataset, sample_indices)
    
    dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size = batch_size,                              
        shuffle = True, 
        num_workers = max(1, multiprocessing.cpu_count() - 1)
    )
    
    return dataloader


# dataloader for CIFAR-10
def loader_cifar(dir_data, train = False, batch_size = 250):
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    if train:
        trainset = torchvision.datasets.CIFAR10(root = dir_data, train = True, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = max(1, multiprocessing.cpu_count() - 1))
    else:
        testset = torchvision.datasets.CIFAR10(root = dir_data, train = False, download = True, transform = transform_test)
        dataloader = torch.utils.data.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = max(1, multiprocessing.cpu_count() - 1))
    return dataloader


def get_data_specs(pretrained_dataset):
    if pretrained_dataset == "imagenet":
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 1000
        input_size = 224
        # input_size = 299 # inception_v3
        num_channels = 3
    elif pretrained_dataset == 'caltech':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        num_classes = 101
        input_size = 224
        num_channels = 3
    elif pretrained_dataset == 'asl':
        mean = [0., 0., 0.]
        std = [1., 1., 1.]
        num_classes = 29
        input_size = 200
        num_channels = 3
    elif pretrained_dataset == 'eurosat':
        mean = [0.3442, 0.3801, 0.4077]
        std = [0.2025, 0.1368, 0.1156]
        num_classes = 10
        input_size = 64
        num_channels = 3
    else:
        raise ValueError
    return num_classes, (mean, std), input_size, num_channels



def fix_labels(test_set):
    val_dict = {}
    groudtruth = os.path.join(IMAGENET_PATH, 'validation/classes.txt')

    i = 0
    with open(groudtruth) as file:
        for line in file:
            (key, class_name) = line.split(':')
            val_dict[key] = i
            i = i + 1

    new_data_samples = []
    for i, j in enumerate(test_set.samples):
        class_id = test_set.samples[i][0].split('/')[-1].split('.')[0].split('_')[-1]
        org_label = val_dict[class_id]
        new_data_samples.append((test_set.samples[i][0], org_label))

    test_set.samples = new_data_samples
    return test_set


def get_data(dataset):
    num_classes, (mean, std), input_size, num_channels = get_data_specs(dataset)

    if dataset == "imagenet":

        traindir = os.path.join(IMAGENET_PATH, 'validation')

        train_transform = transforms.Compose([
            transforms.Resize(256),
            # transforms.Resize(299), # inception_v3
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        full_val = ImageFolder(root=traindir, transform=train_transform)
        full_val = fix_labels(full_val)

        full_index = np.arange(0, len(full_val))
        index_test = np.load(IMAGENET_PATH + '/validation/index_test.npy').astype(np.int64)
        index_train = [x for x in full_index if x not in index_test]
        train_data = torch.utils.data.Subset(full_val, index_train)
        test_data = torch.utils.data.Subset(full_val, index_test)
        #print('test size {} train size {}'.format(len(test_data), len(train_data)))

    elif dataset == 'caltech':
        traindir = os.path.join(CALTECH_PATH, "train")
        testdir = os.path.join(CALTECH_PATH, "test")

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(size=input_size),
            #transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        train_data_full = ImageFolder(root=traindir, transform=train_transform)
        train_data = torch.utils.data.Subset(train_data_full,
                                             np.random.choice(len(train_data_full),
                                                              size=(int(0.5 * len(train_data_full))
                                                                    - int(0.5 * len(train_data_full)) % 100),
                                                              replace=False))
        # train_data = train_data_full
        test_data = ImageFolder(root=testdir, transform=test_transform)
        test_data = torch.utils.data.Subset(test_data,
                                            np.random.choice(len(test_data),
                                                             size=(len(test_data)
                                                                   - len(test_data) % 100),
                                                             replace=False))
        print('[DEBUG] caltech train len: {}'.format(len(train_data_full)))
        print('[DEBUG] caltech test len: {}'.format(len(test_data)))
    elif dataset == 'asl':
        traindir = os.path.join(ASL_PATH, "train")
        testdir = os.path.join(ASL_PATH, "test")

        train_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.RandomCrop(input_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        train_data_full = ImageFolder(root=traindir, transform=train_transform)
        train_data = torch.utils.data.Subset(train_data_full,
                                             np.random.choice(len(train_data_full),
                                                              size=(int(0.5 * len(train_data_full))
                                                                    - int(0.5 * len(train_data_full)) % 100),
                                                              replace=False))
        test_data = ImageFolder(root=testdir, transform=test_transform)
        test_data = torch.utils.data.Subset(test_data,
                                            np.random.choice(len(test_data),
                                                             size=(len(test_data)
                                                                   - len(test_data) % 100),
                                                             replace=False))
        #print('[DEBUG] asl train len: {}'.format(len(train_data)))
        #print('[DEBUG] asl test len: {}'.format(len(test_data)))

    elif dataset == 'eurosat':
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean, std)
        ])

        train_dataset = EuroSAT(transform=train_transform)

        test_dataset = EuroSAT(transform=test_transform)

        trainval, _ = random_split(train_dataset, 0.9, random_state=42)
        train_data_full, _ = random_split(trainval, 0.9, random_state=7)
        train_data = torch.utils.data.Subset(
            train_data_full,
            np.random.choice(len(train_data_full),
                             size=int(0.5 * len(train_data_full) - int(0.5 * len(train_data_full)) % 100),
                             replace=False))
        _, test_data = random_split(test_dataset, 0.9, random_state=42)
        print('[DEBUG] train len: {}'.format(len(train_data)))
        print('[DEBUG] test len: {}'.format(len(test_data)))
    return train_data, test_data


# Evaluate model on data with or without UAP
# Assumes data range is bounded by [0, 1]
def evaluate(model, loader, uap = None, n = 5):
    '''
    OUTPUT
    top         top n predicted labels (default n = 5)
    top_probs   top n probabilities (default n = 5)
    top1acc     array of true/false if true label is in top 1 prediction
    top5acc     array of true/false if true label is in top 5 prediction
    outputs     output labels
    labels      true labels
    '''
    probs, labels = [], []
    model.eval()
    
    if uap is not None:
        _, (x_val, y_val) = next(enumerate(loader))
        batch_size = len(x_val)
        uap = uap.unsqueeze(0).repeat([batch_size, 1, 1, 1])
    
    with torch.set_grad_enabled(False):
        for i, (x_val, y_val) in enumerate(loader):
            
            if uap is None:
                out = torch.nn.functional.softmax(model(x_val.cuda()), dim = 1)
            else:
                perturbed = torch.clamp((x_val + uap).cuda(), 0, 1) # clamp to [0, 1]
                out = torch.nn.functional.softmax(model(perturbed), dim = 1)
                
            probs.append(out.cpu().numpy())
            labels.append(y_val)
            
    # Convert batches to single numpy arrays    
    probs = np.stack([p for l in probs for p in l])
    labels = np.array([t for l in labels for t in l])
    
    # Extract top 5 predictions for each example
    top = np.argpartition(-probs, n, axis = 1)[:,:n]
    top_probs = probs[np.arange(probs.shape[0])[:, None], top].astype(np.float16)
    top1acc = top[range(len(top)), np.argmax(top_probs, axis = 1)] == labels
    top5acc = [labels[i] in row for i, row in enumerate(top)]
    outputs = top[range(len(top)), np.argmax(top_probs, axis = 1)]
        
    return top, top_probs, top1acc, top5acc, outputs, labels
