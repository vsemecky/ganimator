class IDriver:
    """ Interface that every GAN driver must meet. """

    # def __init__(self, path: str):
    #     self.path = path
    #     self.G = None

    def get_z_dim(self):
        """ Return dimension of z latent space """
        raise NotImplementedError()
