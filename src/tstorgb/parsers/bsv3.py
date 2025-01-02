import numpy as np
from pathlib import Path
from pyvips import Image, Interpolate
from tstorgb.parsers.rgb import rgb_parser
from tstorgb.parsers.addons.bsv3_addon import (
    crop_cells,
    get_states,
    frame_iterator,
)


def bsv3_parser(bsv3_file):
    with open(bsv3_file, "rb") as f:
        bsv3_set = set()

        check = int.from_bytes(f.read(2), byteorder="little", signed=False)

        if check in {259, 515}:
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
            cells_imgs, cells_types, bytepos = crop_cells(
                bsv3_file, bytepos=f.tell(), rgb_img=rgb_img, cellnumber=cellnumber
            )

            # Get frames.
            frames, statenames, stateitems = bsv3a(
                bsv3_file, bytepos, cells_imgs, cells_types, is_alpha
            )

            return (frames, statenames, stateitems, bsv3_set, True)

        elif check in {771}:
            # Read cell data.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

            # Get rgb image.
            rgb_img_path = Path(bsv3_file.parent, bsv3_file.stem + ".rgb")
            rgb_img = rgb_parser(rgb_img_path)
            bsv3_set.add(rgb_img_path.name)

            if rgb_img is False:
                return (None, list(), list(), set(), False)

            # Get cells.
            cells_imgs, cells_types, bytepos = crop_cells(
                bsv3_file,
                bytepos=f.tell(),
                rgb_img=rgb_img,
                cellnumber=cellnumber,
            )

            # Get frames.
            frames, statenames, stateitems = bsv3b(
                bsv3_file, bytepos, cells_imgs, cells_types, is_alpha
            )

            return (frames, statenames, stateitems, bsv3_set, True)
        else:
            # Unsupported or invalid file.
            return (None, list(), list(), set(), False)


def bsv3a(bsv3_file, bytepos, cells_imgs, cells_types, is_alpha):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells_imgs = [list() for _ in range(blocks)]

        # Subcell data.
        a = [np.array([]) for _ in range(blocks)]
        b = [np.array([]) for _ in range(blocks)]
        affine_matrix = [np.array([]) for _ in range(blocks)]
        alpha = [np.array([]) for _ in range(blocks)]
        interp = Interpolate.new("bicubic")

        # Read blocks info.
        for i in range(blocks):
            subcellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Ignore extra byte.
            f.seek(f.tell() + 1)

            # Info about each subcell.
            subcells_imgs[i] = [
                Image.new_from_array(np.zeros((1, 1, 4)), interpretation="srgb")
                for _ in range(subcellnumber)
            ]
            a[i] = np.zeros((2, subcellnumber), dtype=np.float32)
            b[i] = np.zeros((2, subcellnumber), dtype=np.float32)
            affine_matrix[i] = np.array(
                [np.identity(2)] * subcellnumber, dtype=np.float32
            )
            alpha[i] = np.zeros(subcellnumber, dtype=int)

            # Skip empty blocks.
            if subcellnumber == 0:
                subcells_imgs[i]
                continue

            # Process subcells.
            for j in range(subcellnumber):
                index = int.from_bytes(f.read(2), byteorder="little", signed=False)

                # Get top left corner.
                a[i][..., j] = np.frombuffer(
                    f.read(8), dtype=np.float32, count=2
                ).transpose()

                # Get coefficients for the affine matrix.
                # [sx rx]
                # [ry sy]
                affine_matrix[i][j, ...] = (
                    np.frombuffer(
                        f.read(16),
                        dtype=np.float32,
                        count=4,
                    )
                    .reshape(2, 2)
                    .transpose()
                )

                # Get alpha.
                if is_alpha == 1:
                    alpha[i][j] = int.from_bytes(
                        f.read(1), byteorder="little", signed=False
                    )
                else:
                    alpha[i][j] = 255

                subcells_imgs[i][j] = cells_imgs[index].copy()
                subcell_canvas = Image.new_from_array(
                    np.zeros(
                        [
                            subcells_imgs[i][j].height + 1,
                            subcells_imgs[i][j].width + 1,
                            4,
                        ]
                    ),
                    interpretation="srgb",
                )
                subcells_imgs[i][j] = subcell_canvas.composite(  # type: ignore
                    subcells_imgs[i][j], x=1, y=1, mode="over", premultiplied=False
                )

                # Resize a little bit cropped cells that are rotated.
                if (
                    affine_matrix[i][j, 1, 0] != 0 or affine_matrix[i][j, 0, 1] != 0
                ) and cells_types[index] == "crop":
                    subcells_imgs[i][j] = subcells_imgs[i][j].resize(
                        1
                        + 4
                        / np.min(
                            [subcells_imgs[i][j].width, subcells_imgs[i][j].height]
                        ),
                        kernel="nearest",
                    )

                subcells_imgs[i][j] *= [1, 1, 1, alpha[i][j] / 255]  # type: ignore
                subcells_imgs[i][j] = subcells_imgs[i][j].affine(  # type: ignore
                    affine_matrix[i][j, ...].flatten().tolist(),
                    interpolate=interp,
                    extend="background",
                )

                # Adjust coordinates.
                if np.linalg.det(affine_matrix[i][j, ...]) > 0:
                    a[i][0, j] = np.round(a[i][0, j])
                    a[i][1, j] = np.floor(a[i][1, j])
                else:
                    a[i][..., j] = np.floor(a[i][..., j])

                # Get bottom right corner.
                b[i][..., j] = a[i][..., j] + np.array(
                    [subcells_imgs[i][j].width, subcells_imgs[i][j].height]
                )

        # Get canvas dimensions.
        c = np.hstack([a[i] for i in range(blocks) if len(subcells_imgs[i]) > 0])
        d = np.hstack([b[i] for i in range(blocks) if len(subcells_imgs[i]) > 0])
        c.sort()
        d.sort()

        canvas_dim = np.array(np.ceil(d[..., -1] - c[..., 0]), dtype=int)
        canvas_img = Image.new_from_array(
            np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
        )

        # Correct coordinates.
        a = [a[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

        # Get states info.
        statenames, stateitems = get_states(bsv3_file, f.tell())

        # Get frames.
        frames = frame_iterator(canvas_img, subcells_imgs, a)
        return (frames, statenames, stateitems)


def bsv3b(bsv3_file, bytepos, cells_imgs, cells_types, is_alpha):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells_imgs = [Image.black(1, 1) for _ in range(subcellnumber)]  # type: ignore

        # Subcell data.
        a = np.zeros((2, subcellnumber), dtype=np.float32)
        b = np.zeros((2, subcellnumber), dtype=np.float32)
        affine_matrix = np.array([np.identity(2)] * subcellnumber, dtype=np.float32)
        alpha = np.zeros(subcellnumber, dtype=int)
        interp = Interpolate.new("bicubic")

        # Process subcells.
        for j in range(subcellnumber):
            index = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Get top left corner.
            a[..., j] = np.frombuffer(f.read(8), dtype=np.float32, count=2).transpose()

            # Get coefficients for the affine matrix.
            # [sx rx]
            # [ry sy]
            affine_matrix[j, ...] = (
                np.frombuffer(
                    f.read(16),
                    dtype=np.float32,
                    count=4,
                )
                .reshape(2, 2)
                .transpose()
            )

            # Get alpha.
            if is_alpha == 1:
                alpha[j] = int.from_bytes(f.read(1), byteorder="little", signed=False)
            else:
                alpha[j] = 255

            subcells_imgs[j] = cells_imgs[index].copy()
            subcell_canvas = Image.new_from_array(
                np.zeros(
                    [
                        subcells_imgs[j].height + 1,
                        subcells_imgs[j].width + 1,
                        4,
                    ]
                ),
                interpretation="srgb",
            )
            subcells_imgs[j] = subcell_canvas.composite(  # type: ignore
                subcells_imgs[j],
                x=1,
                y=1,
                mode="over",
            )

            # Resize a little bit cropped cells that are rotated.
            if (
                affine_matrix[j, 1, 0] != 0 or affine_matrix[j, 0, 1] != 0
            ) and cells_types[index] == "crop":
                subcells_imgs[j] = subcells_imgs[j].resize(
                    1 + 4 / np.min([subcells_imgs[j].width, subcells_imgs[j].height]),
                    kernel="nearest",
                )

            subcells_imgs[j] *= [1, 1, 1, alpha[j] / 255]  # type: ignore
            subcells_imgs[j] = subcells_imgs[j].affine(  # type: ignore
                affine_matrix[j, ...].flatten().tolist(),
                interpolate=interp,
                extend="background",
            )

            # Adjust coordinates.
            if np.linalg.det(affine_matrix[j, ...]) > 0:
                a[0, j] = int(a[0, j] + 0.5)
                a[1, j] = np.floor(a[1, j] + 0.75)
            else:
                a[0, j] = int(a[0, j] + 0.95)
                a[1, j] = np.floor(a[1, j])

            # Get bottom right corner.
            b[..., j] = a[..., j] + np.array(
                [subcells_imgs[j].width, subcells_imgs[j].height]
            )

        # Get canvas dimensions.
        c = a.copy()
        c.sort()
        d = b.copy()
        d.sort()

        canvas_dim = np.array(np.ceil(d[..., -1] - c[..., 0]), dtype=int)
        canvas_img = Image.new_from_array(
            np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
        )

        # Correct coordinates.
        a -= np.array(c[..., 0]).reshape(2, 1)

        # Read frames info.
        frames_items = [list() for _ in range(blocks)]
        fi_a = [np.array([]) for _ in range(blocks)]
        for i in range(blocks):
            subcells = int.from_bytes(f.read(2), byteorder="little", signed=False)
            f.seek(f.tell() + 1)  # extra byte to ignore.

            frames_items[i] = [Image.black(1, 1) for _ in range(subcells)]  # type: ignore
            fi_a[i] = np.zeros((2, subcells))
            for j in range(subcells):
                index = int.from_bytes(f.read(2), byteorder="little", signed=False)
                frames_items[i][j] = subcells_imgs[index].copy()
                fi_a[i][..., j] = a[..., index]

        # Get states info.
        statenames, stateitems = get_states(bsv3_file, f.tell())

        # Get frames.
        frames = frame_iterator(canvas_img, frames_items, fi_a)
        return (frames, statenames, stateitems)
