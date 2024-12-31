import numpy as np
from pathlib import Path
from copy import deepcopy
from importlib import resources
from pyvips import Image, Interpolate
from tstorgb.parsers import rgb, rgb_parser


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

        subcells_img = [None for _ in range(blocks)]
        byte_pos = [0 for _ in range(blocks)]
        a = np.zeros((2, blocks), dtype=int)
        b = np.zeros((2, blocks), dtype=int)
        subcell_dim = np.zeros((2, blocks), dtype=int)

        # Read blocks info.
        for i in range(blocks):
            # Read 24 bit byte position.
            byte_value = int.from_bytes(f.read(1), "little", signed=False)
            byte_pos[i] = (
                int.from_bytes(f.read(2), "little", signed=False) << 8
            ) + byte_value

            # Ignore extra byte.
            f.read(1)

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            a[..., i] = np.frombuffer(f.read(4), dtype=np.int16, count=2).transpose()

            # Ignore extra byte.
            f.read(1)

            # Get the subcell image.
            rgb_img = rgb_parser(bcell_file, byte_seek=byte_pos[i])
            if rgb_img is False:
                continue
            else:
                subcells_img[i] = rgb_img  # type: ignore

            # Get subcell dimensions.
            subcell_dim[0, i] = subcells_img[i].width  # type: ignore
            subcell_dim[1, i] = subcells_img[i].height  # type: ignore

            b[..., i] = a[..., i] + subcell_dim[..., i] + np.array([2, 2])

    c = deepcopy(a)
    d = deepcopy(b)
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)

    # Generate the frames.
    def generate_frames(
        subcells_img,
        blocks,
        a,
        c,
        canvas_dim,
    ):
        for i in range(blocks):
            frame_img = Image.new_from_array(
                np.zeros((canvas_dim[1], canvas_dim[0], 4)),
                interpretation="srgb",
            )

            yield frame_img.composite(  # type: ignore
                subcells_img[i],
                mode="over",
                x=(a[0, i] - c[0, 0]),
                y=(a[1, i] - c[1, 0]),
            )

    frame_iterator = generate_frames(subcells_img, blocks, a, c, canvas_dim)

    return (frame_iterator, blocks, set(), True)


def bcell_11(bcell_file):
    with open(bcell_file, "rb") as f:
        f.seek(8)

        # Set of images used.
        bcell_set = set()

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)

        frames = ["" for _ in range(blocks)]
        subcells_img = [None for _ in range(blocks)]
        a = np.zeros((2, blocks), dtype=int)
        b = np.zeros((2, blocks), dtype=int)
        subcell_dim = np.zeros((2, blocks), dtype=int)

        # Read blocks info.
        for i in range(blocks):
            # Ignore extra byte.
            skip = int.from_bytes(f.read(1))
            frames[i] = f.read(skip).decode("utf8")[0:-1].lower()

            # I'm not sure what this is, I suspect to be the time to wait before showing the next frame.
            # So it gets stored but is unused in this code.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)[0]

            a[..., i] = np.frombuffer(f.read(4), dtype=np.int16, count=2).transpose()

            # Ignore extra byte.
            f.read(1)

            # Get the subcell image.
            rgb_img_path = Path(bcell_file.parent, frames[i])
            rgb_img = rgb_parser(rgb_img_path)
            if rgb_img is False:
                continue
            else:
                bcell_set.add(frames[i])
                subcells_img[i] = rgb_img  # type: ignore

            # Get subcell dimensions.
            subcell_dim[0, i] = subcells_img[i].width  # type: ignore
            subcell_dim[1, i] = subcells_img[i].height  # type: ignore

            b[..., i] = a[..., i] + subcell_dim[..., i] + np.array([2, 2])

    c = deepcopy(a)
    d = deepcopy(b)
    c.sort()
    d.sort()

    canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)

    # Generate the frames.
    def generate_frames(
        subcells_img,
        blocks,
        a,
        c,
        canvas_dim,
    ):
        for i in range(blocks):
            frame_img = Image.new_from_array(
                np.zeros((canvas_dim[1], canvas_dim[0], 4)),
                interpretation="srgb",
            )

            yield frame_img.composite(  # type: ignore
                subcells_img[i],
                mode="over",
                x=(a[0, i] - c[0, 0]),
                y=(a[1, i] - c[1, 0]),
            )

    frame_iterator = generate_frames(subcells_img, blocks, a, c, canvas_dim)

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
        subcells_img = deepcopy(frames)
        a = [np.array([]) for _ in range(blocks)]
        b = deepcopy(a)
        subcell_dim = deepcopy(a)
        affine_matrix = deepcopy(a)
        alpha = deepcopy(a)

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
            subcell_dim[i] = np.zeros((2, subcells[i]), dtype=np.float32)
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
                        subcells_img[i].append(null_img.copy())  # type: ignore
                    else:
                        subcells_img[i].append(rgb_img)

                else:
                    skip = int.from_bytes(f.read(1))
                    frames[i].append(f.read(skip).decode("utf8")[0:-1])

                    skip = int.from_bytes(f.read(1))
                    f.read(skip)

                    # Add shadows.
                    if frames[i][j] == "SHADOW" and not disable_shadows:
                        subcells_img[i].append(shadow_img.copy())  # type: ignore

                    # Anything else is not supported.
                    else:
                        # Create empty image
                        subcells_img[i].append(null_img.copy())  # type: ignore
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

                    alpha[i][j] = np.frombuffer(f.read(4), dtype=np.float32)

                # Get edges in counterclockwise direction.
                edges = (
                    np.array(
                        (subcells_img[i][j].width, subcells_img[i][j].height)
                    ).reshape(2, 1)
                    / 2
                    * np.array([[1, -1, -1, 1], [1, 1, -1, -1]])
                )

                # Transform edges.
                edges = np.matmul(affine_matrix[i][j, ...], edges)
                edges.sort()

                # Get subcell dimensions.
                subcell_dim[i][..., j] = edges[..., -1] - edges[..., 0]

                # Adjust coordinates accordingly to the affine matrix.
                if np.linalg.det(affine_matrix[i][j, ...]) > 0:
                    a[i][0, j] = np.round(a[i][0, j])
                    a[i][1, j] = np.floor(a[i][1, j])
                else:
                    a[i][0, j] = np.floor(a[i][0, j])
                    a[i][1, j] = np.floor(a[i][1, j])

                    # Position shadows correctly according to its determinant signal. [positive -> from left side; negative <- from right side]
                    a[i][0, j] -= subcell_dim[i][0, j]

                # Get bottom right corner.
                b[i][..., j] = a[i][..., j] + subcell_dim[i][..., j] + np.array([2, 2])

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

        canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)

        # Generate the frames.
        def generate_frames(
            subcells_img,
            blocks,
            a,
            c,
            affine_matrix,
            canvas_dim,
        ):
            for i in range(blocks):
                frame_img = Image.new_from_array(
                    np.zeros((canvas_dim[1], canvas_dim[0], 4)),
                    interpretation="srgb",
                )

                for j in range(subcells[i]):
                    subcells_img[i][j] *= [1, 1, 1, alpha[i][j]]
                    subcells_img[i][j] = subcells_img[i][j].affine(
                        affine_matrix[i][j, ...].flatten().tolist(),
                        interpolate=Interpolate.new("bicubic"),
                        extend="background",
                    )

                yield frame_img.composite(  # type: ignore
                    list(reversed(subcells_img[i])),
                    mode="over",
                    x=list(reversed(a[i][0, ...] - c[0, 0])),
                    y=list(reversed(a[i][1, ...] - c[1, 0])),
                )

        frame_iterator = generate_frames(
            subcells_img,
            blocks,
            a,
            c,
            affine_matrix,
            canvas_dim,
        )

        return (frame_iterator, blocks, bcell_set, True)
