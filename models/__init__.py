from .base import *
from .vanilla_vae import *
from .vq_vae import *
from .timbre_vae import *
# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VQVAE':VQVAE,
              'VanillaVAE':VanillaVAE,
              'MusicTimbreVAE': MusicTimbreVAE}
