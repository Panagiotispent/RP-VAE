
''' Activate visdom server first. Go in a shell and write visdom'''


#!/usr/bin/env python3
import argparse
import torch
import torchvision
import utils
import numpy as np
from model_Full import VAE # model_Full model_D model_RP model_RP_D
from data import TRAIN_DATASETS, DATASET_CONFIGS
from train import train_model
from Test import test_model

torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser('VAE PyTorch implementation')
parser.add_argument('--dataset', default='cifar10',
                    choices=list(TRAIN_DATASETS.keys()))

parser.add_argument('--kernel-num', type=int, default=750)
parser.add_argument('--z-size', type=int, default= 10 )

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--sample-size', type=int, default=20)
parser.add_argument('--lr', type=float, default=3e-05)
parser.add_argument('--weight-decay', type=float, default=1e-06)

parser.add_argument('--loss-log-interval', type=int, default=100)
parser.add_argument('--image-log-interval', type=int, default=500)
parser.add_argument('--resume', action='store_true')#,default = True)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')


## set default= True to train or test within spyder
main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--test', action='store_false', dest='train')#,default=False)   
main_command.add_argument('--train', action='store_true',default =True)


if __name__ == '__main__':
    args = parser.parse_args()
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    dataset = TRAIN_DATASETS[args.dataset]

    vae = VAE(
        label=args.dataset,
        image_size=dataset_config['size'],
        channel_num=dataset_config['channels'],
        kernel_num=args.kernel_num,
        z_size=args.z_size,
    )
    
    # move the model parameters to the gpu if needed.
    if cuda:
        vae.cuda()
    print('Train: ', args.train)
    #split data train/test 
    #Reproducibility 
    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)
    train_set, val_set = torch.utils.data.random_split(dataset, [40000, 10000],generator=g_cpu)
    
    
    
    # run a test or a training process.
    if args.train:
        train_model(
            vae, dataset=train_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=args.checkpoint_dir,
            loss_log_interval=args.loss_log_interval,
            image_log_interval=args.image_log_interval,
            resume=args.resume,
            cuda=cuda,
        )
    else:
        #Monte Carlo runs
        loss_lst = []
        for i in range(10):
            loss = test_model(
                vae, dataset=val_set,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                checkpoint_dir=args.checkpoint_dir,
                loss_log_interval=args.loss_log_interval,
                image_log_interval=args.image_log_interval,
                resume=args.resume,
                cuda=cuda,
            )
            
            loss_lst.append(loss)
        mean = np.mean(loss_lst)
        sd = np.std(loss_lst)    
        images = vae.sample(args.sample_size)
        torchvision.utils.save_image(images, args.sample_dir+'/'+vae.name+'test.png')
