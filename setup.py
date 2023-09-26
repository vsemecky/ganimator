import setuptools

requirements = open('requirements.txt').read().splitlines()

setuptools.setup(
    name='ganimator',
    version='0.0.2',
    author='Vojtěch Semecký',
    author_email='vojtech@semecky.cz',
    description='GAN Animator - library to generate videos and images from GAN neural networks',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/vsemecky/ganimator',
    license='MIT',
    packages=setuptools.find_packages(),
    package_data = {"ganimator": ["dnnlib/**/*", "torch_utils/**/*"]},
    include_package_data=True,
    install_requires=requirements,
)
