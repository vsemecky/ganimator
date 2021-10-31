from typing import Tuple
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from . import IDriver


class StyleGanDriver(IDriver):
    """
    This driver connects Ganimator to StyleGan networks based on the original Nvidia implementations.

    Supported networks:
        - StyleGan, StyleGan2, StyleGan2-Ada, StyleGan3
        - both Tensorflow and PyTorch implementations
        - including StyleGan trained in RunwayML
        - including non-square networks trained using SkyFlyNill/StyleGan2 or RoyWheels/StyleGan2-Ada
    """

    def __init__(self, pkl: str, cache_dir: str = None):
        """
        Loads network into memory
        :type path: str Filename or URL
        :type cache_dir: str Local path do cache dir
        """
        self.device = torch.device('cuda')
        print(f'Loading networks from {pkl}')
        with dnnlib.util.open_url(pkl, cache_dir=cache_dir) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)
            self.z_dim = self.G.z_dim
            self.is_conditional = (self.G.c_dim != 0)

        # with open(path, 'rb') as f:
        #     self.G = pickle.load(f)['G_ema'].cuda()  # torch.nn.Module

    def get_z_dim(self) -> int:
        """ Return dimension of z latent space """
        return self.G.z_dim

    def seed_to_z(self, seed: int):
        """ Experiment"""
        # z = torch.randn([seed, self.G.z_dim]).cuda()
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, self.G.z_dim)
        ).to(self.device)

        return z

    @staticmethod
    def _make_transform_matrix(translate: Tuple[float, float], rotate: float):
        """
        Construct an rotation/translation matrix to pass to the generator.
        The generator expects this matrix as an inverse to avoid potentially failing
        numerical operations in the network.
        """
        matrix = np.eye(3)
        # rotate_rad = rotate / 360.0 * np.pi * 2  # Convert degrees to radians
        rotate_rad = rotate * np.pi / 180  # Convert degrees to radians
        sinus = np.sin(rotate_rad)
        cosinus = np.cos(rotate_rad)
        matrix[0][0] = cosinus
        matrix[0][1] = sinus
        matrix[0][2] = translate[0]
        matrix[1][0] = -sinus
        matrix[1][1] = cosinus
        matrix[1][2] = translate[1]

        return np.linalg.inv(matrix)

    def generate_image(self, z=None, label_id=None, trunc=1, translate=(0, 0), rotate=0, noise_mode='const'):
        """
        noise_mode: 'const', 'random', 'none'
        """

        # Labels
        # TODO: Prepare `torch.zeros([1, self.G.c_dim], device=self.device)` in the constructor. Here just copy pepared zeros.
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            label[:, label_id] = 1

        # If applicable, perform the transformation
        if hasattr(self.G.synthesis, 'input'):
            m = self._make_transform_matrix(translate, rotate)
            self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = self.G(z, label, truncation_psi=trunc, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
