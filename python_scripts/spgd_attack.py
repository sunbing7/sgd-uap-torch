import os
import sys
sys.path.append(os.path.realpath('..'))
from attacks import uap_sgd
from utils import evaluate
from utils import *
from collections import OrderedDict
import warnings
warnings.filterwarnings("ignore")

import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description='Trains a sPGD UAP')
    # pretrained
    parser.add_argument('--dataset', default='imagenet', choices=['imagenet', 'caltech', 'asl', 'eurosat'],
                        help='Used dataset to generate UAP (default: imagenet)')

    parser.add_argument('--arch', default='resnet50', choices=['googlenet', 'vgg19', 'resnet50',
                                                               'shufflenetv2', 'mobilenet'],
                        help='Used model architecture: (default: resnet50)')

    parser.add_argument('--epoch', type=int, default=2,
                        help='Number of epoch')

    parser.add_argument('--targets', type=int, nargs="*", default=[0])

    parser.add_argument('--model_name', type=str, default='vgg19_cifar10.pth',
                        help='model name (default: vgg19_cifar10.pth)')

    parser.add_argument('--proj_dir',
                        help='Current dir')

    parser.add_argument('--adaptive_attack', default=0, type=int,
                        help='If conduct adaptive attack')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers (default: 6)')
    args = parser.parse_args()

    return args


def attack(args):
    data_train, _ = get_data(args.dataset)
    train_loader = torch.utils.data.DataLoader(data_train,
                                               batch_size=100,
                                               shuffle=True,
                                               num_workers=4,
                                               pin_memory=True)
    _, data_test = get_data(args.dataset)
    test_loader = torch.utils.data.DataLoader(data_test,
                                              batch_size=100,
                                              shuffle=False,
                                              num_workers=4,
                                              pin_memory=True)

    num_classes, (mean, std), input_size, num_channels = get_data_specs(args.dataset)
    model_weights_path = str(args.proj_dir) + '/models/' + str(args.model_name)
    network = get_network(args.arch, num_classes=num_classes)
    # Set the target model into evaluation mode
    network.eval()

    if args.dataset == "caltech" or args.dataset == 'asl':
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
        else:
            #state dict
            orig_state_dict = torch.load(model_weights_path, map_location=torch.device('cpu'))
            new_state_dict = OrderedDict()
            for k, v in network.state_dict().items():
                if k in orig_state_dict.keys():
                    new_state_dict[k] = orig_state_dict[k]

            network.load_state_dict(new_state_dict)

    elif args.dataset == 'eurosat':
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))

    elif args.dataset == "imagenet" and 'repaired' in args.model_name:
        network = torch.load(model_weights_path, map_location=torch.device('cpu'))
    elif args.dataset == "cifar10":
        if 'repaired' in args.model_name:
            network = torch.load(model_weights_path, map_location=torch.device('cpu'))
            adaptive = '_adaptive'
        else:
            network.load_state_dict(torch.load(model_weights_path, map_location=torch.device('cpu')))
            if 'trades' in args.model_name:
                adaptive = '_trades'

    network = nn.DataParallel(network)

    # Normalization wrapper, so that we don't have to normalize adversarial perturbations
    normalize = Normalizer(mean=mean, std=std)
    network = nn.Sequential(normalize, network)

    network = network.cuda()

    # clean accuracy
    #_, _, _, _, outputs, labels = evaluate(network, test_loader)
    #print('Accuracy:', sum(outputs == labels) / len(labels

    nb_epoch = args.epoch
    eps = 10 / 255
    beta = 12
    step_decay = 0.6

    for y_target in args.targets:
        print('Target class {}'.format(y_target))
        uap, losses = uap_sgd(network, train_loader, nb_epoch, eps, beta, step_decay, y_target=y_target)

        # evaluate
        _, _, _, _, outputs, labels = evaluate(network, test_loader, uap = uap)
        print('Accuracy:', sum(outputs == labels) / len(labels))
        print('Targeted success rate:', sum(outputs == y_target) / len(labels))

        if args.adaptive_attack:
            torch.save(
                uap, '../uaps/' + str(args.arch) + '_' + str(args.dataset) + '/uap_' + str(y_target) + 'adaptive.pth')
        else:
            torch.save(uap, '../uaps/' + str(args.arch) + '_' + str(args.dataset) + '/uap_' + str(y_target) + '.pth')


if __name__ == '__main__':
    args = parse_arguments()
    attack(args)
