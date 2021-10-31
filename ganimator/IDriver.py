import PIL
import numpy as np
from typing import Tuple


class IDriver:
    """ Interface that every GAN driver must meet. """

    def __init__(self, path: str):
        self.path = path
        self.z_dim = None
        raise NotImplementedError()

    def seed_to_z(self, seed: int):
        """ Converts seed to vector in z latent space """
        z_np = np.random.RandomState(seed).randn(1, self.z_dim)
        return z_np

    def generate_image(
            self,
            z: np.ndarray,
            label_id=None,
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),
            rotate: float = 0,
            noise_mode='const'  # 'const', 'random', 'none'
    ) -> PIL.Image:
        raise NotImplementedError()
