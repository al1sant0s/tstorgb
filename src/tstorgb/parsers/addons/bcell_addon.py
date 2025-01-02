import numpy as np
from pyvips import Image


def generate_frames(
    canvas_img,
    subcells_imgs,
    a,
):
    for i, subcells in zip(range(len(subcells_imgs)), subcells_imgs):
        yield canvas_img.composite(
            list(reversed(subcells)),
            mode="over",
            x=list(reversed(np.array(a[i][0, ...], dtype=int))),
            y=list(reversed(np.array(a[i][1, ...], dtype=int))),
            premultiplied=False,
        )
