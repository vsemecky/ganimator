import pickle
import numpy as np
from . import IDriver

# import sys
# sys.path.append("./submodules/stylegan2-ada/")

import dnnlib
import dnnlib.tflib as tflib


class StyleGanTfDriver(IDriver):
    """
    The driver connects Ganimator to TensorFlow versions of StyleGan

    Supported networks:
        - StyleGan, StyleGan2, StyleGan2-Ada
        - including StyleGan/StyleGan2 trained in RunwayML
        - including non-square networks trained using SkyFlyNill/StyleGan2 or RoyWheels/StyleGan2-Ada
    """

    def __init__(self, path: str, cache_dir: str = None):
        """
        Loads network into memory
        :type path: str Path to pkl file (local file or URL)
        """
        dnnlib.tflib.init_tf()

        self.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        # with open(path, 'rb') as stream:
        with dnnlib.util.open_url(path, cache_dir=cache_dir) as stream:
            print(f'Loading network from {path}')
            _G, _D, self.Gs = pickle.load(stream, encoding='latin1')

        super().__init__(path=path, z_dim=self.Gs.z_dim, c_dim=self.Gs.c_dim)

    def generate_image(
            self,
            z: np.ndarray = None,
            trunc: float = 1,
            noise_mode: str = 'const',  # 'const', 'random', 'none'
            **kwargs
    ):
        tflib.init_tf()
        image_np = self.Gs.run(
            z,
            None,
            truncation_psi=trunc,
            randomize_noise=(noise_mode == 'random'),
            output_transform=self.output_transform
        )[0]

        return image_np
