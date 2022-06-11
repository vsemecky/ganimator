from typing import Tuple
import dnnlib
import numpy as np
import PIL.Image
import torch
import legacy
from . import IDriver


class StyleGanPyDriver(IDriver):
    """
    The driver connects Ganimator to Pytorch versions of StyleGan

    Supported networks:
        - StyleGan2-Ada-Pytorch
        - StyleGan3
        - Legacy support: most of StyleGan2/StyleGan2-Ada networks TensorFlow edition

    Note: Due to legacy support, the driver can also load most of the networks from older Tensorflow StyleGan implementations.
    If you have problem with loading older networks, try StyleGanTfDriver() instead, which should load all networks
    from Tensorflow StyleGan implementations.
    """

    def __init__(self, path: str, cache_dir: str = None, anamorphic: Tuple[int, int] = None):
        """
        Loads network into memory
        :type path: str Filename or URL
        :type cache_dir: str Local path do cache dir
        """
        print(f'Loading networks from {path}')
        self.device = torch.device('cuda')
        self.anamorphic = anamorphic
        with dnnlib.util.open_url(path, cache_dir=cache_dir) as f:
            self.G = legacy.load_network_pkl(f)['G_ema'].to(self.device)

        if anamorphic:
            width, height = anamorphic
        else:
            width = height = self.G.img_resolution

        super().__init__(
            path=path,
            z_dim=self.G.z_dim,
            c_dim=self.G.c_dim,
            width=width,
            height=height,
        )

    def generate_image(
            self,
            z: np.ndarray = None,
            label_id=None,
            trunc: float = 1,
            translate: Tuple[float, float] = (0, 0),  # Only for config-t and config-r. Ignored on other configurations.
            rotate: float = 0,  # Only for config-t and config-r. Ignored on other configurations.
            noise_mode='const',  # 'const', 'random', 'none'
            **kwargs
    ):
        # Labels
        # TODO: Prepare `torch.zeros([1, self.G.c_dim], device=self.device)` in the constructor. Here just copy pepared zeros.
        label = torch.zeros([1, self.G.c_dim], device=self.device)
        if self.G.c_dim != 0:
            label[:, label_id] = 1

        # If applicable perform the translate/rotate transformation
        if hasattr(self.G.synthesis, 'input'):
            self.G.synthesis.input.transform.copy_(torch.from_numpy(
                self._make_transform_matrix(translate, rotate)
            ))

        z = np.expand_dims(z, axis=0)  # shape [512] => [512x1]
        z_tensor = torch.from_numpy(z).to(self.device)  # np.ndarray => torch.Tensor
        img = self.G(z_tensor, label, truncation_psi=trunc, noise_mode=noise_mode)  # Generate image
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)  # Some magic (???)
        img = img[0].cpu().numpy()

        # If anamorphic, apply reverse process (resize image from square to original anamorphic size)
        if self.anamorphic:
            # @todo neprevadet na PIL a zpet, zkusit resize z OpenCv
            # img = cv2.resize(img, dsize=self.anamorphic, interpolation=cv2.INTER_LANCZOS4)
            img_pil = PIL.Image.fromarray(img).resize(self.anamorphic, resample=PIL.Image.LANCZOS)
            img = np.array(img_pil)

        return img

    @staticmethod
    def _make_transform_matrix(translate: Tuple[float, float], rotate: float) -> torch.Tensor:
        """
        Construct a rotation/translation matrix to pass to the generator.
        The generator expects this matrix as an inverse to avoid potentially failing
        numerical operations in the network.
        """
        matrix = np.eye(3)
        rotate_rad = rotate * np.pi / 180  # degrees => radians
        sinus = np.sin(rotate_rad)
        cosinus = np.cos(rotate_rad)
        matrix[0][0] = cosinus
        matrix[0][1] = sinus
        matrix[0][2] = translate[0]
        matrix[1][0] = -sinus
        matrix[1][1] = cosinus
        matrix[1][2] = translate[1]

        return np.linalg.inv(matrix)
