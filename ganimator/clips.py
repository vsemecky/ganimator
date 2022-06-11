import scipy
from .functions import *


class StaticImageClip(ImageClip):
    """ Single static image generated from GAN. This clip will simply display the same image at all times."""

    def __init__(
            self,
            gan: IDriver,
            duration: int = 30,
            seed: int = 42,
            label_id=None,
            trunc: float = 1,
            title=None,
            title_font_size=None,
    ):
        pil_image = generate_image(gan=gan, seed=seed, label_id=label_id, trunc=trunc)

        if title is not None:
            # Append title text
            height = 768  # temporary hack
            title_font_size = title_font_size or height // 32
            title_font = get_image_font(family='sans-serif', weight='normal', size=title_font_size)
            draw = ImageDraw.Draw(pil_image, 'RGBA')  # RGBA because of semitransparent rectangle arount the title
            draw_text(draw=draw, image=pil_image, font=title_font, text=title, gravity="South", fill=(0, 0, 0, 200), margin=height // 64, padding=title_font_size // 5)

        super().__init__(np.array(pil_image), duration=duration)


class LatentWalkClip(VideoClip):
    """ Generates Random latent walk clip """

    def __init__(
            self,
            gan: IDriver,
            duration: int = 30,
            fps: int = 30,
            seed: int = 42,
            label_id=None,
            trunc: float = 1.0,
            smoothing_sec: float = 1.0,
            title=None,
            title_font_size=None
    ):
        num_frames = duration * fps
        random_state = np.random.RandomState(seed)

        # Generating latent vectors
        shape = [num_frames, gan.z_dim]  # [frame, component]
        all_latents = random_state.randn(*shape).astype(np.float32)
        all_latents = scipy.ndimage.gaussian_filter(all_latents, [smoothing_sec * fps] + [0], mode='wrap')
        all_latents /= np.sqrt(np.mean(np.square(all_latents)))

        def make_frame(t):
            """ Frame generation func for MoviePy """
            frame_idx = int(np.clip(np.round(t * fps), 0, num_frames - 1))
            latents = all_latents[frame_idx]
            image_np = gan.generate_image(latents, trunc=trunc, label_id=label_id)

            if title is not None:  # Append title text
                height = image_np.shape[0]
                font_size = title_font_size or height // 32
                title_font = get_image_font(family='sans-serif', weight='normal', size=font_size)
                pil_image = Image.fromarray(image_np)
                draw = ImageDraw.Draw(pil_image, 'RGBA')  # RGBA because of semitransparent rectangle arount the title
                draw_text(draw=draw, image=pil_image, font=title_font, text=title, gravity="South", fill=(0, 0, 0, 200), margin=height // 64, padding=font_size // 5)
                image_np = np.array(pil_image)

            return image_np

        # Create VideoClip
        super().__init__(make_frame=make_frame, duration=duration)


class ArrayClip(CompositeVideoClip):
    """
    Object-oriented implementation of MoviePy function array_clip()

    rows_widths
      widths of the different rows in pixels. If None, is set automatically.

    cols_widths
      widths of the different colums in pixels. If None, is set automatically.

    bg_color
       Fill color for the masked and unfilled regions. Set to None for these
       regions to be transparent (will be slower).

    """
    def __init__(self, array, rows_widths=None, cols_widths=None, bg_color=None):
        array = np.array(array)
        sizes_array = np.array([[c.size for c in line] for line in array])

        # find row width and col_widths automatically if not provided
        if rows_widths is None:
            rows_widths = sizes_array[:, :, 1].max(axis=1)
        if cols_widths is None:
            cols_widths = sizes_array[:, :, 0].max(axis=0)

        xx = np.cumsum([0] + list(cols_widths))
        yy = np.cumsum([0] + list(rows_widths))

        for j, (x, cw) in enumerate(zip(xx[:-1], cols_widths)):
            for i, (y, rw) in enumerate(zip(yy[:-1], rows_widths)):
                clip = array[i, j]
                w, h = clip.size
                if (w < cw) or (h < rw):
                    clip = (CompositeVideoClip([clip.set_position('center')],
                                               size=(cw, rw),
                                               bg_color=bg_color).
                            set_duration(clip.duration))

                array[i, j] = clip.set_position((x, y))

        super().__init__(array.flatten(), size=(xx[-1], yy[-1]), bg_color=bg_color)


class TruncComparisonClip(ArrayClip):
    """  Truncation psi comparison clip """

    def __init__(
            self,
            gan: IDriver,
            grid=(3, 2),  # (cols, rows)
            trunc_range=(0.0, 1.0),  # (trunc_min, trunc_max)
            fps=30,  # frames per second
            duration=30,  # Duration in seconds
            smoothing_sec=1.0,
            seed=420  # Starting seed of the first image
    ):
        cols, rows = grid
        trunc_min, trunc_max = trunc_range
        step = (trunc_max - trunc_min) / (cols * rows - 1)

        # empty array rows x cols
        clips = [[0 for col in range(cols)] for row in range(rows)]

        i = 0
        for row in range(rows):
            for col in range(cols):
                trunc = trunc_min + i * step
                i += 1
                height = 768  # temporary hack (until drivers will be done)
                if trunc == 0:  # Save some time: Truncation psi=0 generates allways the same average image.
                    clips[row][col] = StaticImageClip(
                        gan=gan,
                        seed=seed,
                        trunc=trunc,
                        duration=duration,
                        title="psi 0.0",
                        title_font_size=rows * height // 32,
                    )
                else:
                    clips[row][col] = LatentWalkClip(
                        gan=gan,
                        seed=seed,
                        trunc=trunc,
                        duration=duration,
                        smoothing_sec=smoothing_sec,
                        fps=fps,
                        title="psi " + str(round(trunc, 2)),
                        title_font_size=rows * height // 32,
                    )

        # Arrange clips into ArrayClip (parent class)
        super().__init__(clips)


class ProgressClip(VideoClip):

    def __init__(self, fakes_dir, fps=30, fps_step=None, height=1080, heading="Training Progress", subheading="StyleGAN"):

        VideoClip.__init__(self)

        sequence = sorted(glob.glob(f"{fakes_dir}/fakes??????.jpg"))  # todo: support also png
        self.fps = fps
        width = height * 16 // 9  # Calc `width` to keep ratio 16:9
        self.durations = []
        self.images_starts = []
        pointer = 0
        for i, val in enumerate(sequence):
            if fps_step:  # increasing fps
                fps_moving = min(fps_step * i + 1, fps)
                # print(i, "fps_moving", fps_moving)
                duration = 1.0 / fps_moving
            else:  # fix fps
                duration = 1.0 / fps
            self.durations.append(duration)
            self.images_starts.append(pointer)
            pointer += duration

        self.duration = sum(self.durations)
        self.end = self.duration
        self.sequence = sequence
        self.lastindex = None
        self.lastimage = None

        # Calc text block height
        pil_fake = Image.open(self.sequence[0])  # get test image
        pil_fake.thumbnail(size=(width, height))
        text_block_height = height - pil_fake.height

        # Prepare fonts
        heading_font = ImageFont.truetype(font_by_name('sans-serif', 'bold'), size=text_block_height // 4)
        subhead_font = ImageFont.truetype(font_by_name('sans-serif', 'normal'), size=text_block_height // 6)
        counter_font = ImageFont.truetype(font_by_name('monospace', 'normal'), size=text_block_height // 6)

        # Background image with text headings
        pil_background = Image.new(mode="RGB", size=(width, height), color="black")  # New empty image
        draw = ImageDraw.Draw(pil_background)
        # Heading text
        heading_w, heading_h = draw.textsize(heading, font=heading_font)
        draw.text(xy=((width - heading_w) // 2, text_block_height // 20), text=heading, fill=(255, 255, 255), font=heading_font, align="center")
        # Sub-Heading text
        subh_w, subh_h = draw.textsize(subheading, font=subhead_font)
        draw.text(xy=((width - subh_w) // 2, heading_h + text_block_height // 10), text=subheading, fill=(255, 255, 255), font=subhead_font, align="center")

        def make_frame(t):
            index = max(
                [i for i in range(len(self.sequence)) if self.images_starts[i] <= t]
            )

            if index != self.lastindex:
                pil_image = pil_background.copy()
                pil_fake = Image.open(self.sequence[index])
                pil_fake.thumbnail(size=(width, height))
                pil_image.paste(pil_fake, (0, text_block_height))

                # Counter text [kimg]
                draw = ImageDraw.Draw(pil_image)
                kimg = re.search(r'\d+', self.sequence[index]).group()
                text = f"{kimg} kimg"
                text_w, text_h = draw.textsize(text, font=counter_font)
                draw.text(xy=((width - text_w) // 2, text_block_height - text_h - text_block_height // 10), text=text, fill=(255, 255, 255), font=counter_font, align="center")

                self.lastimage = np.array(pil_image)
                self.lastindex = index
            return self.lastimage  # Prevest na array jen jednou a ulozit do lastimage

        self.make_frame = make_frame
        self.size = make_frame(0).shape[:2][::-1]
