# -*- coding: utf-8 -*-
import argparse

parser = argparse.ArgumentParser(description='Training SNN')
parser.add_argument('--seed', default=200, type=int, help='random seed')#60

# model setting
parser.add_argument('--stu_arch', default="resnet18", type=str,
                    help="resnet18|resnet19")
parser.add_argument('--tea_arch', default="resnet19", type=str, help="resnet34")

parser.add_argument('--tea_path',default='PATH', type=str)

parser.add_argument('--dataset', default="CIFAR100", type=str, help="CIFAR10|CIFAR100")
parser.add_argument('--data_path', default='dataset_PATH', type=str)
parser.add_argument('--log_path', default="PATH", type=str, help="log path")
parser.add_argument('--auto_aug', default=True, action='store_true')#True
parser.add_argument('--cutout', default=True, action='store_true')#True

# learning setting
parser.add_argument('--optim', default='SGDM', type=str)
parser.add_argument('--scheduler', default='COSINE', type=str)
parser.add_argument('--train_batch_size', default=256, type=int)
parser.add_argument('--val_batch_size', default=256, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=5e-4, type=float)
parser.add_argument('--num_epoch', default=300, type=int)#300
parser.add_argument('--num_workers', default=8, type=int)

# spiking neuron setting 
parser.add_argument('--decay', default=0.5, type=float)

parser.add_argument('--thresh', default=1.0, type=float)
parser.add_argument('--T', default=6, type=int, help='num of time steps')
parser.add_argument('--step_mode', default='m', help='step mode')
parser.add_argument('--v_reset', default=0.0, type=float)
parser.add_argument('--detach_reset', default=False, action='store_true')
# training algorithm
parser.add_argument('--device', default='cuda:1', type=str)
parser.add_argument('--alpha', default=0.5, type=float)#2.0
parser.add_argument('--beta', default=0.5, type=float)


parser.add_argument('--error_model', default='step', type=str)
args = parser.parse_args()
