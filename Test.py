
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual
import torchvision
import vis_utils
import torch
avg_meter = vis_utils.AverageMeter()

# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"



def test_model(model, dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                loss_log_interval=30,
                image_log_interval=300,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):
    
    lnplt = vis_utils.VisdomLinePlotter(env_name=model.name) # cause we want the name variable
    
    model.eval()
    
    #load trained model
    utils.load_checkpoint(model, checkpoint_dir)
    

    data_loader = utils.get_data_loader(dataset, batch_size,False, cuda=cuda)
    data_stream = tqdm(enumerate(data_loader, 1))
    
    # # # #Generate projection matrices 
    if ('RP' in model.name) or ('RP_D' in model.name):
    #     # Fixed sampling / Reproducibility 
    #     g = torch.Generator()
        
    #     random_samples = torch.zeros([len(data_loader.dataset),model.z_size, model.cov_space]) # tri.shape >> z_size 
    #     P = torch.zeros([len(data_loader.dataset),model.z_size, model.cov_space])
    #     for i in range(len(data_loader.dataset)):
    #         g.manual_seed(i)
    #         random_samples[i] = torch.randn(model.z_size, model.cov_space, generator=g) 
    #         (P[i],_) = torch.linalg.qr(random_samples[i])
            
        # # Random sampling
        random_samples = torch.randn(model.z_size, model.cov_space).repeat(len(data_loader.dataset), 1, 1)
        (P,_) = torch.linalg.qr(random_samples)
       
    for batch_index, (x, y) in data_stream:

        # prepare data on gpu if needed
        x = Variable(x).cuda() if cuda else Variable(x)
        
        #For the projection matrix
        batch_iter = batch_index * batch_size
        
        if ('RP' in model.name) or ('RP_D' in model.name):
            (mean, ltr, var), x_reconstructed = model(x,P[(batch_iter-batch_size):batch_iter])
        else:
            (mean, ltr, var), x_reconstructed = model(x)
        
        reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
        kl_divergence_loss = model.kl_divergence_loss(mean, ltr,var)
 
        total_loss = reconstruction_loss.cpu() + kl_divergence_loss.cpu()
        
        
        # update progress
        data_stream.set_description((
            'progress: [{tested}/{total}] ({progress:.0f}%) | '
            'Test loss => '
            'total: {total_loss:.4f} / '
            're: {reconstruction_loss:.3f} / '
            'kl: {kl_divergence_loss:.3f}'
        ).format(
           
            tested=batch_index * len(x),
            total=len(data_loader.dataset),
            progress=(100. * batch_index / len(data_loader)),
            total_loss=total_loss.data.item(),
            reconstruction_loss=reconstruction_loss.data.item(),
            kl_divergence_loss=kl_divergence_loss.data.item(),
        ))

       
        # losses = [
        #     reconstruction_loss.data.item(),
        #     kl_divergence_loss.data.item(),
        #     total_loss.data.item(),
        # ]
        # names = ['reconstruction', 'kl divergence', 'total']
        
        
        # visual.visualize_scalars(
        #     losses, names, 'loss', iteration= batch_index,
        #     env=model.name)
                                                           
   
    images = model.sample(sample_size)
    torchvision.utils.save_image(images, './samples/'+model.name+' Test.png')
    visual.visualize_images(
        images, name = 'generated samples',
        label=str(y[:8].numpy()),
        env=model.name
    )# label name's the first 8 samples
    
    
    return total_loss.detach()
    
    