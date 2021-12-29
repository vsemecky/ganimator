# Ganimator
Ganimator (GAN Animator) is python library to generate beautyfull videos from StyleGan networks. It combines the power of **StyleGan** and **MoviePy**.

## Still in development...

# TODO
- driver for TensorFlow versions of Stylegan
- documentation
- Colab Notebook with examples
- InterpolationClip

# Drivers - connecting to GANs

Ganimator has curently two drivers:
- StyleGanTfDriver
- StyleGanPyDriver

Note: Although Ganimator was created mainly for StyleGan, its driver-based architecture supports connection to any image GAN in the future.
# Clips
|                         |StyleGanPy|StyleGanTf|
|-------------------------|:--------:|:--------:|
|StaticImageClip          | &#10004; | TODO     |
|LatentWalkClip           | &#10004; | TODO     |
|InterpolationClip        | TODO     | TODO     |
|TruncComparisonClip      | &#10004; | TODO     |
