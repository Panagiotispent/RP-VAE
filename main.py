
# ''' -Not required- Activate visdom server first. Go in a shell and write visdom'''


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
parser.add_argument('--dataset', default='mnist',
                    choices=list(TRAIN_DATASETS.keys()))

parser.add_argument('--model', type=str, default='model_Flow') # model_Full model_D model_RP model_RP_D, model_Flow
parser.add_argument('--flow-type', type=str, default='bnaf') # planar, radial, householder, bnaf

parser.add_argument('--kernel-num', type=int, default=500)
parser.add_argument('--z-size', type=int, default= 10 )
parser.add_argument('--n-size', type=int, default= 20 ) # either RP lower full rank dimensions (eg z//10), or bnaf linear matrices dimensions (eg. n = 2z) 

parser.add_argument('--f-len', type=int, default= 5 )  # 32 for flows, way less flow length (eg 4) for bnaf 
parser.add_argument('--f-layers', type=int, default= 2 )
parser.add_argument('--res', type=str, default='None') # None, normal, gated

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
main_command = parser.add_mutually_exclusive_group(required=False)
# main_command.add_argument('--test', action='store_false', dest='train',default=False)   
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
    elif args.model == 'model_RP_D':
        from model_RP_D import VAE
        
    elif args.model == 'model_Flow':
        from model_Flow import VAE
        
    print(args.model)
    
    cuda = args.cuda and torch.cuda.is_available()
    dataset_config = DATASET_CONFIGS[args.dataset]
    train_set = TRAIN_DATASETS[args.dataset]
    test_set = TEST_DATASETS[args.dataset]
    
    if 'Flow' in args.model:             
        vae = VAE(
           label=args.dataset + ' ' +args.flow_type ,
           image_size=dataset_config['size'],
           channel_num=dataset_config['channels'],
           kernel_num=args.kernel_num,
           z_size=args.z_size,
           flow = args.flow_type,
           length = args.f_len,
           f_layers = args.f_layers,
           n_size=args.n_size,
           res = args.res
       )
    else:
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
        