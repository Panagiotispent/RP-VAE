from torch import optim
from torch.autograd import Variable
from tqdm import tqdm
import utils
import visual
import torchvision
import vis_utils
import torch
avg_meter = vis_utils.AverageMeter()
# lnplt = vis_utils.VisdomLinePlotter()

#for visualisation of higher latent space to 2D purposes only 

from sklearn.decomposition import PCA
pca = PCA(n_components=2)


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
    
    lnplt = vis_utils.VisdomLinePlotter(env_name=model.name) # cause we want the name variable
    
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

    for epoch in range(epoch_start, epochs+1):
        data_loader = utils.get_data_loader(dataset, batch_size,False, cuda=cuda)
        data_stream = tqdm(enumerate(data_loader, 1))
        
        model.seed = 0 # for the RP
        
        for batch_index, (x, y) in data_stream:
            # where are we?
            iteration = (epoch-1)*(len(dataset)//batch_size) + batch_index

            # prepare data on gpu if needed
            x = Variable(x).cuda() if cuda else Variable(x)
            
            # flush gradients and run the model forward
            optimizer.zero_grad()
            (mean, ltr, var), x_reconstructed = model(x)
            reconstruction_loss = model.reconstruction_loss(x_reconstructed, x)
            kl_divergence_loss = model.kl_divergence_loss(mean,ltr, var)
 
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

            if iteration % loss_log_interval == 0:
                losses = [
                    reconstruction_loss.data.item(),
                    kl_divergence_loss.data.item(),
                    total_loss.data.item(),
                ]
                names = ['reconstruction', 'kl divergence', 'total']
                lnplt.plot(names[0], 'train', 'reconstruction_loss', epoch, losses[0])
                lnplt.plot(names[1], 'train', 'kl_divergence_loss', epoch, losses[1])
                lnplt.plot(names[2], 'train', 'total_loss', epoch, losses[2])
                
                # visual.visualize_scalars(
                #     losses, names, 'loss',
                #     iteration, env=model.name)
                                                               
            if iteration % image_log_interval == 0:
                to_vis = model.get_z().cpu().numpy()
                #used to train pca to the same dim reduction distribution
                # input(to_vis.shape)
                g_cpu = torch.Generator()
                g_cpu.manual_seed(0)
                fixed_pca = torch.randn(to_vis.shape,generator=g_cpu)
                vis_x = pca.fit(fixed_pca).transform(to_vis.tolist())
                
                visual._vis().scatter(vis_x, opts={'textlabels' : y.cpu().numpy().tolist(), 'title': 'Epoch: ' + str(epoch) },name= str(epoch))        
                images = model.sample(sample_size)
                torchvision.utils.save_image(images, './samples/img'+str(epoch)+'.png')
                visual.visualize_images(
                    images, name = 'generated samples',
                    label=str(y[:8].numpy()),
                    env=model.name
                )# label name's the first 8 samples

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
