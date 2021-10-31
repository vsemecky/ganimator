import pickle
import torch
import torch_utils
import os
import re
from typing import List, Optional, Tuple, Union
import click
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
        """
        print(f'Loading networks from {pkl}')
        self.device = torch.device('cuda')
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
        z = torch.from_numpy(np.random.RandomState(seed).randn(seed, self.G.z_dim)).to(self.device)
        return z

    @staticmethod
    def _make_transform_matrix(translate: Tuple[float, float], rotate: float):
        """
        Construct an inverse rotation/translation matrix to pass to the generator.
        The generator expects this matrix as an inverse to avoid potentially failing
        numerical operations in the network.
        """
        m = np.eye(3)
        s = np.sin(rotate / 360.0 * np.pi * 2)
        c = np.cos(rotate / 360.0 * np.pi * 2)
        m[0][0] = c
        m[0][1] = s
        m[0][2] = translate[0]
        m[1][0] = -s
        m[1][1] = c
        m[1][2] = translate[1]

        return np.linalg.inv(m)

    def generate_image(self, z=None, class_idx=None, truncation_psi=1, translate=(0, 0), rotate=0, noise_mode='const'):
        """
        noise_mode: 'const', 'random', 'none'
        """

        # Labels
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            if class_idx is None:
                raise click.ClickException('Must specify class label with `class_idx` when using a conditional network')
            label[:, class_idx] = 1
        else:
            if class_idx is not None:
                print('Warning: `class_idx` ignored when running on an unconditional network')

        # If applicable, perform the transformation
        if hasattr(self.G.synthesis, 'input'):
            m = self._make_transform_matrix(translate, rotate)
            self.G.synthesis.input.transform.copy_(torch.from_numpy(m))

        img = self.G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB')
