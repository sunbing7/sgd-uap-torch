
##################################################################################################################################
#resnet50 imagenet
#python spgd_attack.py --dataset=imagenet --arch=resnet50 --targets 611 734 854 859 497 577 8 5 --model_name=imagenet_resnet50.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=imagenet --arch=resnet50 --targets 611 --model_name=imagenet_resnet50.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

##################################################################################################################################
#googlenet imagenet 901, 581, 20, 700, 671, 83, 138, 802, 197, 619#
python spgd_attack.py --dataset=imagenet --arch=googlenet --targets 901 581 20 700 671 83 138 802 197 619 --model_name=imagenet_googlenet.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=imagenet --arch=googlenet --epoch=5 --targets 901 --model_name=imagenet_googlenet.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0
##################################################################################################################################
#vgg19 imagenet 898, 895, 861, 965, 764, 701, 222, 545, 720, 71
#python spgd_attack.py --dataset=imagenet --arch=vgg19 --targets 898 895 861 965 764 701 222 545 720 71 --model_name=imagenet_vgg19.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=imagenet --arch=vgg19 --targets 898 --model_name=imagenet_vgg19.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

##################################################################################################################################
#mobilenet ASL
#python spgd_attack.py --dataset=asl --arch=mobilenet --targets 3 8 17 28 0 19 18 1 15 14 --model_name=mobilenet_asl.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=asl --arch=mobilenet --epoch=1 --targets 3 --model_name=mobilenet_asl.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0
##################################################################################################################################
#shufflenetv2 caltech
python spgd_attack.py --dataset=caltech --arch=shufflenetv2 --targets 49 56 91 95 92 48 50 3 24 34 --model_name=shufflenetv2_caltech.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=caltech --arch=shufflenetv2 --targets 49 --model_name=shufflenetv2_caltech.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0
##################################################################################################################################
#resnet50 eurosat
#python spgd_attack.py --dataset=eurosat --arch=resnet50 --targets 7 0 1 4 5 9 3 8 2 6 --model_name=resnet50_eurosat.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

#python spgd_attack.py --dataset=eurosat --arch=resnet50 --targets 7 --model_name=resnet50_eurosat.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

##################################################################################################################################
#wideresnet cifar10
python spgd_attack.py --dataset=cifar10 --arch=resnet50 --targets 0 1 2 4 5 6 7 9 --model_name=wideresnet_cifar10.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0

python spgd_attack.py --dataset=cifar10 --arch=resnet50 --targets 7 --model_name=wideresnet_cifar10.pth --proj_dir='/root/autodl-tmp/sunbing/workspace/uap/sgd-uap-torch' --adaptive_attack=0



