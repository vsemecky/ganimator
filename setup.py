import setuptools

requirements = open('requirements.txt').read().splitlines()

setuptools.setup(
    name='ganimator',
    version='0.0.1',
    author='Vojtěch Semecký',
    author_email='vojtech@semecky.cz',
    description='GAN Animator - library to generate videos and images from GAN neural networks',
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url='https://github.com/vsemecky/ganimator',
    license='MIT',
    packages=['ganimator'],
    install_requires=requirements,
)
