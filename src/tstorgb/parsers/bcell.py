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
        elif check == "bcell12":
            return bcell_13(bcell_file, is_alpha = False, disable_shadows=kwargs["disable_shadows"])
        elif check == "bcell13":
            return bcell_13(bcell_file, is_alpha = True, disable_shadows=kwargs["disable_shadows"])
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
        tlc = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        brc = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
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

            tlc[i][..., 0] = np.frombuffer(
                f.read(4), dtype=np.int16, count=2
            ).transpose()

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

            brc[i][..., 0] = tlc[i][..., 0] + subcells_dim[i][..., 0]

    c = deepcopy(np.hstack(tlc))
    d = deepcopy(np.hstack(brc))
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)
    canvas_img = Image.new_from_array(
        np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
    )

    # Correct coordinates.
    tlc = [tlc[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

    # Generate the frames.
    frame_iterator = generate_frames(canvas_img, subcells_imgs, tlc)

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
        tlc = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        brc = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]
        subcells_dim = [np.zeros((2, 1), dtype=int) for _ in range(blocks)]

        # Fallback image.
        null_img = Image.new_from_array(np.zeros((1, 1, 4)), interpretation="srgb")

        # Read blocks info.
        for i in range(blocks):
            # Ignore extra byte.
            skip = int.from_bytes(f.read(1))
            frames[i] = f.read(skip).decode("utf8")[0:-1].lower()

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            tlc[i][..., 0] = np.frombuffer(
                f.read(4), dtype=np.int16, count=2
            ).transpose()

            # Ignore extra byte.
            f.read(1)

            # Get the subcell image.
            rgb_img_path = Path(bcell_file.parent, frames[i])
            rgb_img = rgb_parser(rgb_img_path)
            if rgb_img is False:
                subcells_imgs[i].append(null_img)
                continue
            else:
                bcell_set.add(frames[i])
                subcells_imgs[i].append(rgb_img)

            # Get subcell dimensions.
            subcells_dim[i][0, 0] = subcells_imgs[i][0].width
            subcells_dim[i][1, 0] = subcells_imgs[i][0].height

            brc[i][..., 0] = tlc[i][..., 0] + subcells_dim[i][..., 0]

    # Check if at least a single image was found. If not, then give up.
    if len(bcell_set) == 0:
        return (None, 0, set(), False)

    c = deepcopy(np.hstack(tlc))
    d = deepcopy(np.hstack(brc))
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)
    canvas_img = Image.new_from_array(
        np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
    )

    # Correct coordinates.
    tlc = [tlc[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

    # Generate the frames.
    frame_iterator = generate_frames(canvas_img, subcells_imgs, tlc)

    return (frame_iterator, blocks, bcell_set, True)


def bcell_13(bcell_file, is_alpha, disable_shadows=False):
    with open(bcell_file, "rb") as f:
        f.seek(8)

        # Preload accessories
        shadow_img = Image.new_from_array(
            np.load(resources.open_binary("tstorgb", "accessories", "shadow.npy")),
            interpretation="srgb",
        )
        # Fallback image.
        null_img = Image.new_from_array(np.zeros((1, 1, 4)), interpretation="srgb")

        # Set of images used.
        bcell_set = set()

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells = [0 for _ in range(blocks)]  # Number of subcells.

        # frame data.
        frames = [list() for _ in range(blocks)]
        subcells_imgs = deepcopy(frames)
        tlc = [np.array([]) for _ in range(blocks)]
        brc = deepcopy(tlc)
        subcells_dim = deepcopy(tlc)
        affine_matrix = deepcopy(tlc)
        alpha = deepcopy(tlc)
        interp = Interpolate.new("bicubic")

        # Read blocks info.
        for i in range(blocks):
            # Read subcell.
            skip = int.from_bytes(f.read(1))
            frames[i].append(f.read(skip).decode("utf8")[0:-1].lower())
            bcell_set.add(frames[i][0])
            start = f.tell()

            f.seek(start + 8)
            subcells[i] = int.from_bytes(f.read(2)) + 1
            f.seek(start)

            # Shape the arrays.
            tlc[i] = np.zeros((2, subcells[i]), dtype=np.float32)
            brc[i] = np.zeros((2, subcells[i]), dtype=np.float32)
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
                    tlc[i][..., 0] = np.frombuffer(
                        f.read(4), dtype=np.int16, count=2
                    ).transpose()
                    f.read(2)  # It's already been read.

                    # Get the subcell image.
                    rgb_img = rgb_parser(Path(bcell_file.parent, frames[i][j]))
                    if rgb_img is False:
                        subcells_imgs[i].append(null_img)  # type: ignore
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
                    tlc[i][..., j] = np.frombuffer(
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
                    if is_alpha is True:
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
                    tlc[i][0, j] = np.round(tlc[i][0, j])
                    tlc[i][1, j] = np.floor(tlc[i][1, j])
                else:
                    tlc[i][0, j] = np.floor(tlc[i][0, j])
                    tlc[i][1, j] = np.floor(tlc[i][1, j])

                    # Position shadows correctly according to its determinant signal. [positive -> from left side; negative <- from right side]
                    tlc[i][0, j] -= subcells_dim[i][0, j]

                # Get bottom right corner.
                brc[i][..., j] = tlc[i][..., j] + subcells_dim[i][..., j]

        # Get canvas dimensions.
        c = np.array(
            [
                tlc[i][..., j]
                for i in range(blocks)
                for j in range(subcells[i])
                if frames[i][j] != "null"
            ]
        ).transpose()
        d = np.array(
            [
                brc[i][..., j]
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
        tlc = [tlc[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

        # Generate the frames.
        frame_iterator = generate_frames(canvas_img, subcells_imgs, tlc)

        return (frame_iterator, blocks, bcell_set, True)
