# Random Projection Integration with Vanilla VAE
This repository extends a [vanilla VAE implementation](https://github.com/kuc2477/pytorch-vae) by integrating Random Projection (RP) with minor modifications. The project maintains compatibility with the original codebase's requirements and execution procedures.

Additional Features
Constrained Covariance Matrices: Implementation details can be found in the [Unscented Autoencoder repository](https://github.com/boschresearch/unscented-autoencoder).
Normalizing Flows Comparison: Code references include:
[Dynamic Flow VAE](https://github.com/fmu2/flow-VAE/blob/master/dynamic_flow_vae.py)
[BNAF](https://github.com/nicola-decao/BNAF/blob/master/bnaf.py)

# **Arguments**
| Argument                | Default Value         | Description                                         |
|-------------------------|-----------------------|-----------------------------------------------------|
| `--dataset`             | `'Flowers102'`       | Dataset to use.                                     |
| `--model`               | `'model_D'`          | Model type: `model_Full`, `model_D`, `model_RP`, `model_RP_D`, `model_Flow`. |
| `--kernel-num`          | `500`                | Kernel count for Random Projection.                |
| `--z-size`              | `10`                 | Latent dimension size.                              |
| `--n-size`              | `10`                 | Noise dimension size.                               |
| `--epochs`              | `30`                 | Number of training epochs.                          |
| `--batch-size`          | `100`                | Batch size for training.                            |
| `--sample-size`         | `20`                 | Sample size for image generation.                   |
| `--lr`                  | `3e-05`              | Learning rate.                                      |
| `--weight-decay`        | `1e-06`              | Weight decay for regularization.                    |
| `--loss-log-interval`   | `100`                | Interval for logging loss metrics.                 |
| `--image-log-interval`  | `500`                | Interval for saving image samples.                 |
| `--resume`              | `False`              | Resume training from a checkpoint.                  |
| `--checkpoint-dir`      | `'./checkpoints'`    | Directory to save checkpoints.                      |
| `--sample-dir`          | `'./samples'`        | Directory to save generated samples.                |
| `--no-gpus`             | `False`              | Disable GPU usage.                                  |

### Mutually Exclusive Flags

| Argument   | Default Value | Description       |
|------------|---------------|-------------------|
| `--test`   | `False`       | Run testing.      |
| `--train`  | `True`        | Run training.     |


# **Mutually Exclusive Flags**:
--test (default: False): Run testing.
--train (default: True): Run training.
# **Outputs**
Runtime Metrics: Training runtime and loss metrics.
Graphs and Samples: Generated image samples saved in the samples folder.
# **Notes**
Variants such as Random Projection with block matrices (RP_B, RP_B_lamda) are not integrated.
Uncomment specific sections in the code to enable Visdom visualization.

The image samples are saved in the samples folder.
