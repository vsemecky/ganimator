from typing import Tuple
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from . import StyleGanPyDriver


class StyleGanPyExperimentalDriver(StyleGanPyDriver):
    """
    Experiment
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
            raise ValueError("Only 'config-t' a 'config-r' are supported by StyleGanPyExperimentalDriver.")

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

        if self.ratio > 1:
            image_np = np.concatenate((
                self.generate_subimage(z_tensor, translate=(0.5, 0), label_tensor=label, trunc=trunc, rotate=rotate, noise=noise_mode),
                self.generate_subimage(z_tensor, translate=(-0.5, 0), label_tensor=label, trunc=trunc, rotate=rotate, noise=noise_mode)
            ), axis=1)
        else:
            image_np = np.concatenate((
                self.generate_subimage(z_tensor, translate=(0, 0.5), label_tensor=label, trunc=trunc, rotate=rotate, noise=noise_mode),
                self.generate_subimage(z_tensor, translate=(0, -0.5), label_tensor=label, trunc=trunc, rotate=rotate, noise=noise_mode)
            ), axis=0)

        # Crop
        image_np = self.np_center_crop(image_np)

        return image_np

    def np_center_crop(self, img_np):
        height, width, channel = img_np.shape

        if self.ratio > 1:
            new_width = 2 * int(height * self.ratio / 2)  # round to even
            new_height = height
        else:
            new_width = width
            new_height = 2 * int(width / self.ratio / 2)  # round to even

        x = width // 2 - (new_width // 2)
        y = height // 2 - (new_height // 2)
        return img_np[y: y + new_height, x: x + new_width]

    def generate_subimage(
            self,
            z_tensor,
            label_tensor,
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),
            rotate: float = 0,
            noise='const'
    ):
        self.G.synthesis.input.transform.copy_(torch.from_numpy(
            self._make_transform_matrix(translate, rotate)
        ))
        img = self.G(z_tensor, label_tensor, truncation_psi=trunc, noise_mode=noise)  # Generate image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Some magic (???)
        return img[0].cpu().numpy()
