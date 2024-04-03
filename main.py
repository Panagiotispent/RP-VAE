
# ''' Activate visdom server first. Go in a shell and write visdom'''


#!/usr/bin/env python3
import argparse
import os
import torch
import torchvision
import numpy as np
# from model_Full import VAE # model_Full model_D model_RP
from data import TRAIN_DATASETS, DATASET_CONFIGS, TEST_DATASETS
from train import train_model
from Test import test_model
# print('Num_threads:',torch.get_num_threads())
# torch.set_num_threads(1)
# torch.autograd.set_detect_anomaly(True)


parser = argparse.ArgumentParser('VAE PyTorch implementation')
parser.add_argument('--dataset', default='Flowers102',
                    choices=list(TRAIN_DATASETS.keys()))

parser.add_argument('--model', type=str, default='model_D') # model_Full model_D model_RP

parser.add_argument('--kernel-num', type=int, default=500)
parser.add_argument('--z-size', type=int, default= 10 )
parser.add_argument('--n-size', type=int, default= 10 )

parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch-size', type=int, default=100)
parser.add_argument('--sample-size', type=int, default=20)
parser.add_argument('--lr', type=float, default=3e-05)
parser.add_argument('--weight-decay', type=float, default=1e-06)

parser.add_argument('--loss-log-interval', type=int, default=100)
parser.add_argument('--image-log-interval', type=int, default=500)
parser.add_argument('--resume', action='store_true',default = False)
parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints')
parser.add_argument('--sample-dir', type=str, default='./samples')
parser.add_argument('--no-gpus', action='store_false', dest='cuda')


## set default= True to train or test within spyder
main_command = parser.add_mutually_exclusive_group(required=True)
main_command.add_argument('--test', action='store_false', dest='train',default=False)   
main_command.add_argument('--train', action='store_true',default =True)


if __name__ == '__main__':
    args = parser.parse_args()
    
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir)
    
    if args.model == 'model_Full':
        from model_Full import VAE
    elif args.model == 'model_D':
        from model_D import VAE
    elif args.model == 'model_RP':
        from model_RP import VAE
    
    print(args.model)
    
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_set = TRAIN_DATASETS[args.dataset]
    test_set = TEST_DATASETS[args.dataset]
    
    vae = VAE(
        label=args.dataset,
        image_size=dataset_config['size'],
        channel_num=dataset_config['channels'],
        kernel_num=args.kernel_num,
        z_size=args.z_size,
        n_size=args.n_size,
    )
    
    # move the model parameters to the gpu if needed.
    if cuda:
        vae.cuda()
    print('Train: ', args.train)
    #split data train/test 
    #Reproducibility 
    g_cpu = torch.Generator()
    g_cpu.manual_seed(0)
    
    # run a test or a training process.
    if args.train:
        train_model(
            vae, dataset=train_set,
            epochs=args.epochs,
            batch_size=args.batch_size,
            sample_size=args.sample_size,
            lr=args.lr,
            weight_decay=args.weight_decay,
            checkpoint_dir=str(args.checkpoint_dir+'/'+args.model),
            loss_log_interval=args.loss_log_interval,
            image_log_interval=args.image_log_interval,
            resume=args.resume,
            cuda=cuda,
        )
    else:
        #Monte Carlo runs
        loss_lst = []
        for i in range(5):
            loss = test_model(
                vae, dataset=test_set,
                epochs=args.epochs,
                batch_size=args.batch_size,
                sample_size=args.sample_size,
                lr=args.lr,
                weight_decay=args.weight_decay,
                checkpoint_dir=str(args.checkpoint_dir+'/'+args.model),
                loss_log_interval=args.loss_log_interval,
                image_log_interval=args.image_log_interval,
                resume=args.resume,
                cuda=cuda,
            )
            
            loss_lst.append(loss)
        
        images = vae.sample(args.sample_size)
        torchvision.utils.save_image(images, args.sample_dir+'/'+vae.name+'test.png')
    
        mean = np.mean(loss_lst)
        sd = np.std(loss_lst)  
        print(vae.name)  
        print('MC Mean test loss',mean)    
        print('MC Mean sd loss',sd)