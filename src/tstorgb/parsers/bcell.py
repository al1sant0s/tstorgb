import numpy as np
from pathlib import Path
from copy import deepcopy
from importlib import resources
from pyvips import Image, Interpolate
from tstorgb.parsers import rgb_parser
from tstorgb.parsers.addons.bcell_addon import generate_frames


def bcell_parser(bcell_file, **kwargs):
    with open(bcell_file, "rb") as f:
        check = f.read(7).decode("utf8")
        if check == "bcell10":
            return bcell_10(bcell_file)
        elif check == "bcell11":
            return bcell_11(bcell_file)
        elif check == "bcell13":
            return bcell_13(bcell_file, disable_shadows=kwargs["disable_shadows"])
        else:
            # Unsupported or invalid file.
            return (None, 0, set(), False)


def bcell_10(bcell_file):
    with open(bcell_file, "rb") as f:
        f.seek(7)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)

        subcells_imgs = [list() for _ in range(blocks)]
        bytepos = [0 for _ in range(blocks)]
        a = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        b = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        subcells_dim = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]

        # Read blocks info.
        for i in range(blocks):
            # Read 24 bit byte position.
            bytevalue = int.from_bytes(f.read(1), "little", signed=False)
            bytepos[i] = (
                int.from_bytes(f.read(2), "little", signed=False) << 8
            ) + bytevalue

            # Ignore extra byte.
            f.read(1)

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            a[i][..., 0] = np.frombuffer(f.read(4), dtype=np.int16, count=2).transpose()

            # Ignore extra byte.
            f.read(1)

            # Get the subcell image.
            rgb_img = rgb_parser(bcell_file, byte_seek=bytepos[i])
            if rgb_img is False:
                continue
            else:
                subcells_imgs[i].append(rgb_img)

            # Get subcell dimensions.
            subcells_dim[i][0, 0] = subcells_imgs[i][0].width
            subcells_dim[i][1, 0] = subcells_imgs[i][0].height

            b[i][..., 0] = a[i][..., 0] + subcells_dim[i][..., 0]

    c = deepcopy(np.hstack(a))
    d = deepcopy(np.hstack(b))
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)
    canvas_img = Image.new_from_array(
        np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
    )

    # Correct coordinates.
    a = [a[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

    # Generate the frames.
    frame_iterator = generate_frames(canvas_img, subcells_imgs, a)

    return (frame_iterator, blocks, set(), True)


def bcell_11(bcell_file):
    with open(bcell_file, "rb") as f:
        f.seek(8)

        # Set of images used.
        bcell_set = set()

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)

        frames = ["" for _ in range(blocks)]
        subcells_imgs = [list() for _ in range(blocks)]
        a = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        b = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        subcells_dim = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]

        # Read blocks info.
        for i in range(blocks):
            # Ignore extra byte.
            skip = int.from_bytes(f.read(1))
            frames[i] = f.read(skip).decode("utf8")[0:-1].lower()

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            a[i][..., 0] = np.frombuffer(f.read(4), dtype=np.int16, count=2).transpose()

            # Ignore extra byte.
            f.read(1)

            # Get the subcell image.
            rgb_img_path = Path(bcell_file.parent, frames[i])
            rgb_img = rgb_parser(rgb_img_path)
            if rgb_img is False:
                continue
            else:
                bcell_set.add(frames[i])
                subcells_imgs[i].append(rgb_img)

            # Get subcell dimensions.
            subcells_dim[i][0, 0] = subcells_imgs[i][0].width
            subcells_dim[i][1, 0] = subcells_imgs[i][0].height

            b[i][..., 0] = a[i][..., 0] + subcells_dim[i][..., 0]

    c = deepcopy(np.hstack(a))
    d = deepcopy(np.hstack(b))
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)
    canvas_img = Image.new_from_array(
        np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
    )

    # Correct coordinates.
    a = [a[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

    # Generate the frames.
    frame_iterator = generate_frames(canvas_img, subcells_imgs, a)

    return (frame_iterator, blocks, bcell_set, True)


def bcell_13(bcell_file, disable_shadows=False):
    with open(bcell_file, "rb") as f:
        f.seek(8)

        # Preload accessories
        shadow_img = Image.new_from_array(
            np.load(resources.open_binary("tstorgb", "accessories", "shadow.npy")),
            interpretation="srgb",
        )
        null_img = Image.new_from_array(np.zeros((1, 1, 4)), interpretation="srgb")

        # Set of images used.
        bcell_set = set()

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells = [0 for _ in range(blocks)]  # Number of subcells.

        # frame data.
        frames = [list() for _ in range(blocks)]
        subcells_imgs = deepcopy(frames)
        a = [np.array([]) for _ in range(blocks)]
        b = deepcopy(a)
        subcells_dim = deepcopy(a)
        affine_matrix = deepcopy(a)
        alpha = deepcopy(a)
        interp = Interpolate.new("bicubic")

        # Read blocks info.
        for i in range(blocks):
            # Read subcell.
            skip = int.from_bytes(f.read(1))
            frames[i].append(f.read(skip).decode("utf8")[0:-1].lower())
            bcell_set.add(frames[i][0])
            start = f.tell()

            f.seek(start + 9)
            subcells[i] = int.from_bytes(f.read(1)) + 1
            f.seek(start)

            # Shape the arrays.
            a[i] = np.zeros((2, subcells[i]), dtype=np.float32)
            b[i] = np.zeros((2, subcells[i]), dtype=np.float32)
            subcells_dim[i] = np.zeros((2, subcells[i]), dtype=np.float32)
            affine_matrix[i] = np.array(
                [np.identity(2)] * subcells[i], dtype=np.float32
            )
            alpha[i] = np.ones(subcells[i])

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            # Get the other subcells data.
            for j in range(subcells[i]):
                if j == 0:
                    # Get first subcell data.
                    a[i][..., 0] = np.frombuffer(
                        f.read(4), dtype=np.int16, count=2
                    ).transpose()
                    f.read(2)  # It's already been read.

                    # Get the subcell image.
                    rgb_img = rgb_parser(Path(bcell_file.parent, frames[i][j]))
                    if rgb_img is False:
                        subcells_imgs[i].append(null_img.copy())  # type: ignore
                    else:
                        subcells_imgs[i].append(rgb_img)

                else:
                    skip = int.from_bytes(f.read(1))
                    frames[i].append(f.read(skip).decode("utf8")[0:-1])

                    skip = int.from_bytes(f.read(1))
                    f.read(skip)

                    # Add shadows.
                    if frames[i][j] == "SHADOW" and not disable_shadows:
                        subcells_imgs[i].append(shadow_img.copy())  # type: ignore

                    # Anything else is not supported.
                    else:
                        # Create empty image
                        subcells_imgs[i].append(null_img.copy())  # type: ignore
                        f.read(28)
                        frames[i][j] = "null"
                        continue

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
                    alpha[i][j] = np.frombuffer(f.read(4), dtype=np.float32)

                subcells_imgs[i][j] *= [1, 1, 1, alpha[i][j]]
                subcells_imgs[i][j] = subcells_imgs[i][j].affine(
                    affine_matrix[i][j, ...].flatten().tolist(),
                    interpolate=interp,
                    extend="background",
                )

                # Get subcell dimensions.
                subcells_dim[i][0, j] = subcells_imgs[i][j].width
                subcells_dim[i][1, j] = subcells_imgs[i][j].height

                # Adjust coordinates accordingly to the affine matrix.
                if np.linalg.det(affine_matrix[i][j, ...]) > 0:
                    a[i][0, j] = np.round(a[i][0, j])
                    a[i][1, j] = np.floor(a[i][1, j])
                else:
                    a[i][0, j] = np.floor(a[i][0, j])
                    a[i][1, j] = np.floor(a[i][1, j])

                    # Position shadows correctly according to its determinant signal. [positive -> from left side; negative <- from right side]
                    a[i][0, j] -= subcells_dim[i][0, j]

                # Get bottom right corner.
                b[i][..., j] = a[i][..., j] + subcells_dim[i][..., j]

        # Get canvas dimensions.
        c = np.array(
            [
                a[i][..., j]
                for i in range(blocks)
                for j in range(subcells[i])
                if frames[i][j] != "null"
            ]
        ).transpose()
        d = np.array(
            [
                b[i][..., j]
                for i in range(blocks)
                for j in range(subcells[i])
                if frames[i][j] != "null"
            ]
        ).transpose()
        c.sort()
        d.sort()

        canvas_dim = np.array(np.ceil(d[..., -1] - c[..., 0]), dtype=int)
        canvas_img = Image.new_from_array(
            np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
        )

        # Correct coordinates.
        a = [a[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

        # Generate the frames.
        frame_iterator = generate_frames(canvas_img, subcells_imgs, a)

        return (frame_iterator, blocks, bcell_set, True)
