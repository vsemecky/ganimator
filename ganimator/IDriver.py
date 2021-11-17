import PIL
import numpy as np
from typing import Tuple


class IDriver:
    """ Interface that every GAN driver must meet. """

    def __init__(
            self,
            path: str,  # Path to network (local file or URL)
            z_dim: int,  # Input (z) vector dimension (e.g. 512 for StyleGan neworks)
            c_dim: int = 0  # Number of conditional labels (0 = non-conditional network)
    ):
        self.path = path
        self.z_dim = z_dim
        self.c_dim = c_dim

    def seed_to_z(self, seed: int):
        """ Converts seed to vector in the z latent space """
        return np.random.RandomState(seed).randn(self.z_dim)

    def generate_image(
            self,
            z: np.ndarray,
            label_id: int = None,  # Label for conditional networks (ignore on non-conditional)
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),  # Ignore if network doesn't support translation
            rotate: float = 0,  # Ignore if network doesn't support rotation
            noise_mode='const',  # 'const', 'random', 'none'
            ** kwargs  # Allow passing additional specific parameters
    ):
        """ Generates image for specified z-vector """
        raise NotImplementedError("Should be implemented.")
