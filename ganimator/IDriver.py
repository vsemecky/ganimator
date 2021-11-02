import PIL
import numpy as np
from typing import Tuple


class IDriver:
    """ Interface that every GAN driver must meet. """

    def __init__(self, path: str, z_dim: int, is_conditional=False):
        self.path = path
        self.z_dim = z_dim
        self.is_conditional = is_conditional

    def seed_to_z(self, seed: int):
        """ Converts seed to vector in the z latent space """
        z_np = np.random.RandomState(seed).randn(self.z_dim)
        return z_np

    def generate_image(
            self,
            z: np.ndarray,
            label_id=None,
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),
            rotate: float = 0,
            noise_mode='const',  # 'const', 'random', 'none'
            ** kwargs
    ):
        raise NotImplementedError("Method hasn't been implemented yet.")
