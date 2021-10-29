import pickle
import torch
import torch_utils
import ganimator


class StyleGanDriver(ganimator.IDriver):
    """
    This driver connects Ganimator to StyleGan networks based on the original Nvidia implementations.

    Supported networks:
        - StyleGan, StyleGan2, StyleGan2-Ada, StyleGan3
        - both Tensorflow and PyTorch implementations
        - including StyleGan trained in RunwayML
        - including non-square networks trained using SkyFlyNill/StyleGan2 or RoyWheels/StyleGan2-Ada
    """

    def __init__(self, path: str):
        """
        Loads network into memory
        :type path: str Filename or URL
        """
        with open(path, 'rb') as f:
            self.device = torch.device('cuda')
            self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    def get_z_dim(self):
        """ Return dimension of z latent space
            @todo Do we need this?
        """
        return self.G.z_dim

    def generate_image(self):
        z = torch.randn([1, self.G.z_dim]).cuda()  # latent codes
        c = None  # class labels (not used in this example)
        img = self.G(z, c)  # NCHW, float32, dynamic range [-1, +1], no truncation
        return img
