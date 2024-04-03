Random Projection integrated with minor changes to the vanilla vae code: https://github.com/kuc2477/pytorch-vae. This code uses the vanilla VAE code as base so the same requirements and execution procedure hold for this code.

The code for constraining the covariance matrices can be found in: https://github.com/boschresearch/unscented-autoencoder

The code can be run with visdom as well after uncommenting the required code.

Instructions:

Run a visdom local host in a shell / not required if not used.

Execute main.py

args: 

--dataset, default='Flowers102'
--model, default='model_D' # model_Full model_D model_RP
--kernel-num, default=500)
--z-size, default= 10
--n-size, default= 10
--epochs, default=30
--batch-size, default=100
--sample-size, default=20
--lr, default=3e-05
--weight-decay, default=1e-06
--loss-log-interval, default=100
--image-log-interval, default=500
--resume,default = False
--checkpoint-dir, default='./checkpoints'
--sample-dir,default='./samples'
--no-gpus, dest='cuda'

Mutually_exclusive:

--test,default=False)  
--train,default =True

Running main.py outputs runtimes and loss metrics, and generates graphs for image samples.

The image samples are saved in the samples folder.
