from typing import Tuple
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from . import StyleGanDriver


class StyleGanDriverFun(StyleGanDriver):
    """
    """

    def __init__(self, path: str, cache_dir: str = None, ratio: float = 1 / 1):
        """
        Loads network into memory
        :type path: str Filename or URL
        :type cache_dir: str Local path do cache dir
        :type ratio: float Target ratio
        """
        self.ratio = ratio
        super().__init__(path=path, cache_dir=cache_dir)
        if not hasattr(self.G.synthesis, 'input'):
            raise ValueError("Only 'config-t' a 'config-r' are supported by StyleGanDriverFun.")

    def generate_image(
            self,
            z: np.ndarray = None,
            label_id=None,
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),
            rotate: float = 0,
            noise_mode='const',
            **kwargs
    ):
        # Labels
        # TODO: Prepare `torch.zeros([1, self.G.c_dim], device=self.device)` in the constructor. Here just copy pepared zeros.
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            label[:, label_id] = 1

        z = np.expand_dims(z, axis=0)  # shape [512] => [512x1]
        z_tensor = torch.from_numpy(z).to(self.device)  # np.ndarray => torch.Tensor

        # Image left
        self.G.synthesis.input.transform.copy_(torch.from_numpy(
            self._make_transform_matrix((-0.5, 0), rotate)
        ))
        img = self.G(z_tensor, label, truncation_psi=trunc, noise_mode=noise_mode)  # Generate image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Some magic (???)
        img_left_np = img[0].cpu().numpy()

        # Image right
        self.G.synthesis.input.transform.copy_(torch.from_numpy(
            self._make_transform_matrix((0.5, 0), rotate)
        ))
        img = self.G(z_tensor, label, truncation_psi=trunc, noise_mode=noise_mode)  # Generate image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Some magic (???)
        img_right_np = img[0].cpu().numpy()

        # Concat images left+right
        return np.concatenate((img_left_np, img_right_np), axis=1)
