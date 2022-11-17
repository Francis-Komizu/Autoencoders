"""
References: https://github.com/auspicious3000/autovc/blob/master/solver_encoder.py
"""

from torch.nn import functional as F


# reconstruction loss / identity mapping loss
def recon_loss(mel_real, mel_reconst):
    return F.mse_loss(mel_real, mel_reconst)


# code semantic loss / content loss
def content_loss(code_real, code_reconst):
    return F.l1_loss(code_real, code_reconst)
