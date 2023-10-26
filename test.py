import argparse
import os
import numpy as np
import random

import torch
from torchvision import datasets
from torchvision import models
from torchvision import transforms

from test_utils import evaluate
from utils import CLIPGuidedClassifier

parser = argparse.ArgumentParser(description='Evaluates model on ImageNet')
parser.add_argument('--data_path', type=str, help='path to ImageNet dataset')
parser.add_argument('--model', type=str, default='resnet50', help='architecture to use (default: resnet50)')
parser.add_argument('--checkpoint_path', type=str, help='path to pretrained checkpoint')
parser.add_argument('--batch_size', type=int, default=256, help='testing batch size')
parser.add_argument('--num_workers', type=int, default=24, help='Number of pre-fetching threads')
parser.add_argument('--seed', type=int, default=10, help='random seed')
parser.add_argument('--CLIP_model', action='store_true', help='use CLIP based classifier')
parser.add_argument('--CLIP_text_path', type=str, default='./imagenet_text_features/gpt3.pth', help='path to CLIP ImageNet text features')
parser.add_argument('--embed_size', type=int, default=512, help='CLIP embedding size')

args = parser.parse_args()


# define number of classes
num_classes = args.embed_size if args.CLIP_model else 1000


# Seed Everything
seed = args.seed

random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


# define imagenet mean and std
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


# define transforms
preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
    ])
test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    preprocess,
])


# define data loader
valdir = os.path.join(args.data_path, 'val')
val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(valdir, test_transform),
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers)


# define model
if args.CLIP_model:
    net = CLIPGuidedClassifier(backbone_type=args.model, embed_size=args.embed_size, CLIP_text_path=args.CLIP_text_path)
else:
    net = models.get_model(args.model, num_classes=num_classes)


# load checkpoint
ckpt = torch.load(args.checkpoint_path)
net.load_state_dict(ckpt['model'])


# use multiple gpus
net = torch.nn.DataParallel(net).cuda()


# run eval
evaluate(net, val_loader)