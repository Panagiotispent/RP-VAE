# RP-VAE

Random Projection integrated with minor changes to the vanilla vae code: https://github.com/kuc2477/pytorch-vae.
This code uses the vanilla VAE code as base so the same requirements and execution procedure hold for this code.

Instead of a single model.py file, the user can import a different parameterisation using the provided model_*.py file within main.py. (Model names are the same as those denoted in the paper)

Instructions:
1. Run a visdom local host in a shell
2. Through main.py
  - Import the desired parameterisation model
    -- Each model requires manual change of the n parameter through its own .py file
  - Decide on the hyperparameters of the model, and number of Monte Carlo runs
  - Train or test

3. Run main.py

Running main.py generates graphs for training loss metrics, and image samples of the last iteration that can be found in the localhost.

The image samples are also saved in the samples folder.
