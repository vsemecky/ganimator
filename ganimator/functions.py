import time
from datetime import datetime
import glob
import re
import numpy as np
import PIL
from PIL import Image, ImageFont, ImageDraw
from matplotlib import font_manager
from moviepy.editor import *
from typing import Tuple

from . import IDriver


#
# Get font path by font-family name
#
def font_by_name(family='sans-serif', weight='normal') -> str:
    return font_manager.findfont(
        font_manager.FontProperties(family=family, weight=weight)
    )


#
# Get PIL ImageFont by options
#
def get_image_font(family='sans-serif', weight='normal', size=12) -> ImageFont:
    font_path = font_manager.findfont(
        font_manager.FontProperties(family=family, weight=weight)
    )
    return ImageFont.truetype(font_path, size=size)


#
# Draw textbox on PIL Image
#
def draw_text(draw: ImageDraw, image: Image, font, text="Text example", gravity="South", fill=(0, 0, 0), padding=5, margin=10):
    text_width, text_height = draw.textsize(text, font=font)
    gravity = gravity.lower()

    if gravity == 'south':
        x = (image.width - text_width) // 2
        y = image.height - text_height - margin - padding
    elif gravity == 'north':
        x = (image.width - text_width) // 2
        y = margin + padding
    elif gravity == 'center':
        x = (image.width - text_width) // 2
        y = (image.height - text_height) // 2
    elif gravity == 'southwest':
        x = margin + padding
        y = image.height - text_height - margin - padding
    elif gravity == 'southeast':
        x = image.width - margin - padding - text_width
        y = image.height - text_height - margin - padding
    elif gravity == 'northwest':
        x = y = margin + padding
    elif gravity == 'northeast':
        x = image.width - margin - padding - text_width
        y = margin + padding
    else:
        x = y = 0

    draw.rectangle((x - padding, y - padding, x + text_width + padding, y + text_height + padding), fill=fill)
    draw.text((x, y), text=text, font=font)


#
# Generates unified video filename based on opional parameters
#
def generate_video_filename(dir=None, dataset=None, timestamp=False, name="video", seed=None, duration=None, trunc=None, pkl=None):
    if dir:
        file_name = f"/{dir}/"
    else:
        file_name = ""

    if dataset:
        file_name += dataset.replace("/", "-")
    if pkl:
        file_name += time.strftime(' - %Y-%m-%d', time.localtime(os.path.getmtime(pkl)))
    if timestamp:
        file_name += datetime.now().strftime(" - %Y-%m-%d %H:%M")
    if name:
        file_name += " - " + name.replace("/", "-")
    if seed:
        file_name += " - seed={}".format(seed)
    if duration:
        file_name += " - {}sec".format(duration)
    if trunc:
        file_name += " - trunc={:03d}".format(int(100 * trunc))
    file_name += ".mp4"  # Append extension

    return file_name


# from https://colab.research.google.com/drive/1ShgW6wohEFQtqs_znMna3dzrcVoABKIH
def generate_zs_from_seeds(seeds, Gs):
    zs = []
    for seed_idx, seed in enumerate(seeds):
        rnd = np.random.RandomState(seed)
        z = rnd.randn(1, *Gs.input_shape[1:])  # [minibatch, component]
        zs.append(z)
    return zs


def line_interpolate(zs, steps):
    out = []
    for i in range(len(zs) - 1):
        for index in range(steps):
            fraction = index / float(steps)
            out.append(zs[i + 1] * fraction + zs[i] * (1 - fraction))
    return out


def generate_image(
        gan: IDriver,
        seed: int = 42,
        label_id=None,
        trunc: float = 1,
        translate: Tuple[float, float] = (0, 0),
        rotate: float = 0,
        noise_mode='const'  # 'const', 'random', 'none'
        ) -> PIL.Image:
    z = gan.seed_to_z(seed)
    image_np = gan.generate_image(z, label_id=label_id, trunc=trunc, translate=translate, rotate=rotate, noise_mode=noise_mode)
    image_pil = PIL.Image.fromarray(image_np, 'RGB')
    return image_pil


# Finds the latest pkl file in the `folder`. Returns tuple (file path, kimg number)
def locate_latest_pkl(folder: str):
    allpickles = sorted(glob.glob(os.path.join(folder, '0*', 'network-*.pkl')))
    latest_pkl = allpickles[-1]
    re_kimg = re.compile('network-snapshot-(\d+).pkl')
    latest_kimg = int(re_kimg.match(os.path.basename(latest_pkl)).group(1))
    return latest_pkl, latest_kimg
