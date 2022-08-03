from .base import *
from .music_vae_flat import MusicVAEFlat
from .music_vae import MusicVAE
from .vanilla_vae import *
from .vq_vae import *
from .timbre_vae import *
from .timbre_transfer import *
# Aliases
VAE = VanillaVAE
GaussianVAE = VanillaVAE

vae_models = {'VQVAE':VQVAE,
              'VanillaVAE':VanillaVAE,
              'MusicVAE': MusicVAE,
              'MusicVAEFlat': MusicVAEFlat,
              'MusicTimbreVAE': MusicTimbreVAE,
              'TimbreTransfer': TimbreTransfer}
