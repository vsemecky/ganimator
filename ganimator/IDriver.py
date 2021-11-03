import PIL
import numpy as np
from typing import Tuple


class IDriver:
    """ Interface that every GAN driver must meet. """

    def __init__(
            self,
            path: str,  # Path to network (local file or URL)
            z_dim: int,  # Input (z) vector dimension (e.g. 512 for StyleGan neworks)
            is_conditional=False  # Indicates whether the network is conditional or not
    ):
        self.path = path
        self.z_dim = z_dim
        self.is_conditional = is_conditional

    def seed_to_z(self, seed: int):
        """ Converts seed to vector in the z latent space """
        return np.random.RandomState(seed).randn(self.z_dim)

    def generate_image(
            self,
            z: np.ndarray,
            label_id=None,  # Label for conditional networks (ignored on non-conditional)
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),
            rotate: float = 0,
            noise_mode='const',  # 'const', 'random', 'none'
            ** kwargs  # Allow passing additional specific parameters
    ):
        """ Generates image for specified z-vector """
        raise NotImplementedError("Should be implemented.")
