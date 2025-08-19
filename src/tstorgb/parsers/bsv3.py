import numpy as np
from pathlib import Path
from pyvips import Image
from tstorgb.parsers.rgb import rgb_parser
from tstorgb.parsers.addons.bsv3_addon import (
    crop_cells,
    get_frama_data,
    get_states,
    frame_iterator,
)


def bsv3_parser(bsv3_file, subsample_factor):
    with open(bsv3_file, "rb") as f:
        bsv3_set = set()

        check = int.from_bytes(f.read(2), byteorder="little", signed=False)

        if check in {259, 515, 771}:

            # Unknown data.
            if check == 771:
                unknown = np.frombuffer(f.read(4), dtype=np.float32)

            # Read cell data.
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

            # Get rgb image.
            rgb_img_path = Path(bsv3_file.parent, bsv3_file.stem + ".rgb")
            if check == 515:
                skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
                rgb_img_path = Path(
                    bsv3_file.parent, f.read(skip).decode("utf8")[0:-1].lower()
                )

            rgb_img = rgb_parser(rgb_img_path)
            bsv3_set.add(rgb_img_path.name)

            if rgb_img is False:
                return (None, list(), list(), set(), False)

            # Get cells.
            cells_imgs, bytepos = crop_cells(
                bsv3_file, bytepos=f.tell(), rgb_img=rgb_img, cellnumber=cellnumber, subsample_factor=subsample_factor
            )
            f.seek(bytepos)

            # Number of blocks to read.
            blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
            used_blocks = (1 if check == 771 else blocks)

            # Get base frame data used to build the frames later.
            canvas_dim, subcells_imgs, tlc, bytepos = get_frama_data(
                bsv3_file, f.tell(), cells_imgs, is_alpha, subsample_factor, blocks = used_blocks, check = check
            )
            f.seek(bytepos)

            # Get frames
            frames = None
            if check == 771:

                # Get frames items and build frames data from base frame data.
                frames_items = [list() for _ in range(blocks)]
                frames_items_tlc = [np.array([]) for _ in range(blocks)]
                for i in range(blocks):
                    subcells = int.from_bytes(f.read(2), byteorder="little", signed=False)
                    f.seek(f.tell() + 1)  # extra byte to ignore.

                    frames_items[i] = [Image.black(1, 1) for _ in range(subcells)]  # type: ignore
                    frames_items_tlc[i] = np.zeros((2, subcells))
                    for j in range(subcells):
                        index = int.from_bytes(f.read(2), byteorder="little", signed=False)
                        frames_items[i][j] = subcells_imgs[0][index]
                        frames_items_tlc[i][..., j] = tlc[0][..., index]

                # Build frames from gathered data.
                frames = frame_iterator(
                    canvas_dim, frames_items, frames_items_tlc, subsample_factor
                )
            else:
                # Build frames from gathered data.
                frames = frame_iterator(
                    canvas_dim, subcells_imgs, tlc, subsample_factor
                )

            # Get states info.
            statenames, stateitems = get_states(bsv3_file, f.tell())

            return (frames, statenames, stateitems, bsv3_set, True)

        else:
            # Unsupported or invalid file.
            return (None, list(), list(), set(), False)
