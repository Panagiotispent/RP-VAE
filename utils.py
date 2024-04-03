import os
import os.path
import torch
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, train, cuda=False):
    if train:
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=False,
            **({'num_workers': 1, 'pin_memory': True} if cuda else {})
            )
    else:
        return DataLoader(
         dataset, batch_size=batch_size, shuffle=False,
         **({'num_workers': 1, 'pin_memory': True} if cuda else {})
     )


def save_checkpoint(model, model_dir, epoch):
    path = os.path.join(model_dir, model.name)

    # save the checkpoint.
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    torch.save({'state': model.state_dict(), 'epoch': epoch}, path)

    # notify that we successfully saved the checkpoint.
    print('=> saved the model {name} to {path}'.format(
        name=model.name, path=path
    ))


def load_checkpoint(model, model_dir):
    path = os.path.join(model_dir, model.name)

    # load the checkpoint.
    checkpoint = torch.load(path)
    print('=> loaded checkpoint of {name} from {path}'.format(
        name=model.name, path=(path)
    ))

    # load parameters and return the checkpoint's epoch and precision.
    model.load_state_dict(checkpoint['state'])
    epoch = checkpoint['epoch']
    return epoch
