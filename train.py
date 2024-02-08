from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
# import visual
import torchvision
# import vis_utils
import torch
import timeit

# avg_meter = vis_utils.AverageMeter()
# lnplt = vis_utils.VisdomLinePlotter()

# using this because I got multiple versions of openmp on my program 
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


def train_model(model, dataset, epochs=10,
                batch_size=32, sample_size=32,
                lr=3e-04, weight_decay=1e-5,
                loss_log_interval=30,
                image_log_interval=300,
                checkpoint_dir='./checkpoints',
                resume=False,
                cuda=False):
    
    # lnplt = vis_utils.VisdomLinePlotter(env_name=model.name) # cause we want the name variable
    
    
    # prepare optimizer and model
    model.train()
    optimizer = optim.Adam(
        model.parameters(), lr=lr,
        weight_decay=weight_decay,
    )

    if resume:
        epoch_start = utils.load_checkpoint(model, checkpoint_dir)
    else:
        epoch_start = 1
        
    data_loader = utils.get_data_loader(dataset, batch_size,False, cuda=cuda) # data NOT shuffled 

    #  #Generate projection matrices
    if ('RP_B' in model.name):
        # Fixed sampling
        g = torch.Generator()
        max_pos = (model.z_size - model.cov_space)
        
        P = torch.zeros([len(data_loader.dataset),1])
        #Fixed block position 
        for i in range(len(data_loader.dataset)):
            g.manual_seed(i)
            P[i] = torch.randint(0, max_pos,(1,),generator = g) 
    
    elif ('RP' in model.name) or ('RP_D' in model.name):
        # Fixed sampling
        g = torch.Generator()
        
        random_samples = torch.zeros([len(data_loader.dataset),model.z_size, model.cov_space]) # tri.shape >> z_size 
        P = torch.zeros([len(data_loader.dataset),model.z_size, model.cov_space])
        for i in range(len(data_loader.dataset)):
            g.manual_seed(i)
            random_samples[i] = torch.randn(model.z_size, model.cov_space, generator=g) # 
            (P[i],_) = torch.linalg.qr(random_samples[i])
        
        
    for epoch in range(epoch_start, epochs+1):
        
        data_stream = tqdm(enumerate(data_loader, 1))

        for batch_index, (x, y) in data_stream:

            # where are we?
            iteration = (epoch-1)*(len(dataset)//batch_size) + batch_index

            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)
            
            # flush gradients and run the model forward
            optimizer.zero_grad()
            #For the projection matrix
            batch_iter = batch_index * batch_size
            
            if ('RP' in model.name) or ('RP_D' in model.name):
                (mean, ltr, var), x_reconstructed = model(x,P[(batch_iter-batch_size):batch_iter])
                
                if epoch == 1 and batch_index == 1:
                    print('first batch time')
                    print(min(timeit.repeat(lambda: model(x,P[(batch_iter-batch_size):batch_iter]),globals=globals(),number= 100,repeat=10)))
                    
                    print('Each operation time:')
                    model.time_forward(x,P[(batch_iter-batch_size):batch_iter])
                
            else:
                (mean, ltr, var), x_reconstructed = model(x)
                
                if epoch == 1 and batch_index == 1:
                    print('first batch time')
                    print(min(timeit.repeat(lambda: model(x),globals=globals(),number= 100,repeat=10)))
                    
                    print('Each operation time:')
                    model.time_forward(x)
                
                
            
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean,ltr, var)
            
            if epoch == 1 and batch_index == 1:
                print('recon_loss')
                print(min(timeit.repeat(lambda: model.reconstruction_loss(x_reconstructed, x),globals=globals(),number= 100,repeat=10)))
                print('kl_div')
                print(min(timeit.repeat(lambda: model.kl_divergence_loss(mean,ltr, var),globals=globals(),number= 100,repeat=10)))
            
 
            total_loss = reconstruction_loss.cpu() + kl_divergence_loss.cpu()
            # backprop gradients from the loss
            total_loss.backward()
            optimizer.step()

            # update progress
            data_stream.set_description((
                'epoch: {epoch} | '
                'iteration: {iteration} | '
                'progress: [{trained}/{total}] ({progress:.0f}%) | '
                'loss => '
                'total: {total_loss:.4f} / '
                're: {reconstruction_loss:.3f} / '
                'kl: {kl_divergence_loss:.3f}'
            ).format(
                epoch=epoch,
                iteration=iteration,
                trained=batch_index * len(x),
                total=len(data_loader.dataset),
                progress=(100. * batch_index / len(data_loader)),
                total_loss=total_loss.data.item(),
                reconstruction_loss=reconstruction_loss.data.item(),
                kl_divergence_loss=kl_divergence_loss.data.item(),
            ))

            # if iteration % loss_log_interval == 0:
            #     losses = [
            #         reconstruction_loss.data.item(),
            #         kl_divergence_loss.data.item(),
            #         total_loss.data.item(),
            #     ]
            #     names = ['reconstruction', 'kl divergence', 'total']
            #     # lnplt.plot(names[0], 'train', 'reconstruction_loss', epoch, losses[0])
            #     # lnplt.plot(names[1], 'train', 'kl_divergence_loss', epoch, losses[1])
            #     # lnplt.plot(names[2], 'train', 'total_loss', epoch, losses[2])
                
            #     # visual.visualize_scalars(
            #     #     losses, names, 'loss',
            #     #     iteration, env=model.name)
                                                               
            # if iteration % image_log_interval == 0:
                      
            #     images = model.sample(sample_size)
            #     torchvision.utils.save_image(images, './samples/img'+str(epoch)+'.png')
            #     visual.visualize_images(
            #         images, name = 'generated samples',
            #         label=str(y[:8].numpy()),
            #         env=model.name
            #     )# label name's the first 8 samples
                
                

        # notify that we've reached to a new checkpoint.
        print()
        print()
        print('#############')
        print('# checkpoint!')
        print('#############')
        print()

        # save the checkpoint.
        utils.save_checkpoint(model, checkpoint_dir, epoch)
        print()
