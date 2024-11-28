import numpy as np
from copy import deepcopy
from wand.image import Image


def report_progress(prefix_str, parsing_info):
    # Clear line.
    print(150 * "", end="\r")
    # Print progress.
    print(f"{prefix_str} {parsing_info}", end="\r")


def rgb_parser(file, show_progress=False, prefix_str=""):
    if show_progress is True:
        report_progress(prefix_str, "")

    with open(file, "rb") as f:
        f.seek(2)
        check = int.from_bytes(f.read(2), byteorder="little")

        if check == 8192:
            depth = 4
        elif check == 0:
            depth = 8
        else:
            # Unsupported signature,
            return False

        width = int.from_bytes(f.read(2), byteorder="little")
        height = int.from_bytes(f.read(2), byteorder="little")

        buffer = f.read()

    # Get base image
    with Image() as rgb_img:
        with Image(
            blob=buffer,
            width=width,
            height=height,
            depth=depth,
            format="rgba",
        ) as img:
            # Swap collor channels for 4 bit depth.
            if depth == 4:
                rgb_img.image_add(img.channel_images["blue"])  # type: ignore
                rgb_img.image_add(img.channel_images["alpha"])  # type: ignore
                rgb_img.image_add(img.channel_images["red"])  # type: ignore
                rgb_img.image_add(img.channel_images["green"])  # type: ignore
                rgb_img.combine(colorspace="srgb")
            # For 8 bit depth this is not necessary.
            elif depth == 8:
                rgb_img.image_add(img)

        # Revert alpha premultiply.
        rgb_img.image_set(rgb_img.fx("u/u.a", channel="rgb"))

        return rgb_img.clone()


def bsv3_parser(bsv3_file, rgb_img, show_progress=False, prefix_str=""):
    with open(bsv3_file, "rb") as f:
        check = int.from_bytes(f.read(2), byteorder="little", signed=False)
        if check == 259:
            return bsv3_259(bsv3_file, rgb_img, show_progress, prefix_str)
        elif check == 771:
            return bsv3_771(bsv3_file, rgb_img, show_progress, prefix_str)
        else:
            print("Invalid bsv3 signature. Skipping this file.")
            return False


def bsv3_259(bsv3_file, rgb_img, show_progress=False, prefix_str=""):
    with open(bsv3_file, "rb") as f:
        f.seek(2)
        cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
        is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

        cell = np.array([np.zeros(4)] * cellnumber, dtype=int)

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            f.seek(f.tell() + skip)

            # Read x, y, width and height.
            cell[i] = np.frombuffer(f.read(8), dtype=np.uint16)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells = list()  # Number of subcells.

        # Subcell data.
        index = list()
        a = list()
        b = list()
        affine_matrix = list()
        alpha = list()

        # Read blocks info.
        for i in range(blocks):
            if show_progress is True:
                report_progress(prefix_str, f"[{0}/{i + 1}]")

            subcells.append(int.from_bytes(f.read(2), byteorder="little", signed=False))

            # Ignore extra byte.
            f.seek(f.tell() + 1)

            # Skip empty subcells
            if subcells[i] == 0:
                index.append(None)
                a.append(None)
                b.append(None)
                affine_matrix.append(None)
                alpha.append(None)
                continue

            # Retrieve info about each subcell.
            index.append(np.zeros(subcells[i], dtype=int))
            a.append(np.zeros((2, subcells[i]), dtype=int))
            b.append(np.zeros((2, subcells[i]), dtype=int))
            affine_matrix.append(np.array([np.identity(3)] * subcells[i]))  # type: ignore
            alpha.append(np.zeros(subcells[i], dtype=int))

            for j in range(subcells[i]):
                index[i][j] = int.from_bytes(
                    f.read(2), byteorder="little", signed=False
                )

                # Get top left corner.
                a[i][..., j] = np.frombuffer(
                    f.read(8), dtype=np.float32, count=2
                ).transpose()

                # Get coefficients for the ImageMagick distort matrix
                # [sx rx 0]
                # [ry sy 0]
                # [0 0 1]
                affine_matrix[i][j, 0:2, 0:2] = np.frombuffer(
                    f.read(16),
                    dtype=np.float32,
                    count=4,
                ).reshape(2, 2)

                # Get edges in counterclockwise direction.
                edges = (
                    cell[index[i][j]][2:4].reshape(2, 1)
                    / 2
                    * np.array([[1, -1, -1, 1], [1, 1, -1, -1]])
                )

                # Transform edges.
                edges = np.matmul(affine_matrix[i][j, 0:2, 0:2], edges)
                edges.sort()

                # Get bottom right corner.
                b[i][..., j] = (
                    a[i][..., j] + edges[..., -1] - edges[..., 0] + np.array([2, 2])
                )

                # Get alpha.
                if is_alpha == 1:
                    alpha[i][j] = int.from_bytes(
                        f.read(1), byteorder="little", signed=False
                    )
                else:
                    alpha[i][j] = 255

        # Get canvas dimensions.
        c = np.hstack([k for k in a if k is not None])
        c.sort()
        d = np.hstack([k for k in b if k is not None])
        d.sort()

        canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)

        # Get states info.
        statenumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
        statenames = ["" for _ in range(statenumber)]
        stateitems = [0 for _ in range(statenumber)]

        for s in range(statenumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)

            statenames[s] = f.read(skip - 1).decode("utf8")
            f.seek(f.tell() + 1)

            start = int.from_bytes(f.read(2), byteorder="little", signed=False)
            end = int.from_bytes(f.read(2), byteorder="little", signed=False)
            stateitems[s] = end - start + 1

        # Generate the frames.
        def generate_frames(
            rgb_img,
            cell,
            blocks,
            index,
            a,
            c,
            affine_matrix,
            canvas_dim,
            show_progress,
            prefix_str,
        ):
            for i in range(blocks):
                with Image(
                    width=canvas_dim[0], height=canvas_dim[1], background="transparent"
                ) as frame_img:
                    if show_progress is True:
                        report_progress(prefix_str, f"[{i + 1}/{blocks}]")

                    for j in reversed(range(subcells[i])):
                        # Crop the cell.
                        with rgb_img.clone() as subcell:
                            subcell.crop(
                                cell[index[i][j]][0],
                                cell[index[i][j]][1],
                                width=cell[index[i][j]][2],
                                height=cell[index[i][j]][3],
                            )

                            subcell.image_set(subcell.fx(f"u * {alpha[i][j]}/255"))
                            subcell.virtual_pixel = "transparent"
                            subcell.distort(
                                "affine_projection",
                                affine_matrix[i][j, 0:3, 0:2].flatten().tolist(),
                                best_fit=True,
                            )

                            # Get coordinates.
                            coords = a[i][..., j] - c[..., 0]

                            # Set composition.
                            frame_img.composite(
                                image=subcell,
                                left=coords[0],
                                top=coords[1],
                            )
                    yield frame_img

        frame_iterator = generate_frames(
            rgb_img,
            cell,
            blocks,
            index,
            a,
            c,
            affine_matrix,
            canvas_dim,
            show_progress,
            prefix_str,
        )

        return [frame_iterator, statenames, stateitems]


def bsv3_771(bsv3_file, rgb_img, show_progress=False, prefix_str=""):
    with open(bsv3_file, "rb") as f:
        f.seek(2)

        unknown = np.frombuffer(f.read(4), dtype=np.float32)
        cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
        is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

        cell = np.array([np.zeros(4)] * cellnumber, dtype=int)

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            f.seek(f.tell() + skip)

            # Read x, y, width and height.
            cell[i] = np.frombuffer(f.read(8), dtype=np.uint16)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcells = int.from_bytes(
            f.read(2), byteorder="little", signed=False
        )  # Number of subcells.

        # Subcell data.
        index = np.zeros(subcells, dtype=int)
        a = np.zeros((2, subcells), dtype=int)
        b = np.zeros((2, subcells), dtype=int)
        affine_matrix = np.array([np.identity(3)] * subcells)
        alpha = np.zeros(subcells, dtype=int)

        for j in range(subcells):
            index[j] = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Get top left corner.
            a[..., j] = np.frombuffer(f.read(8), dtype=np.float32, count=2).transpose()

            # Get coefficients for the ImageMagick distort matrix
            # [sx rx 0]
            # [ry sy 0]
            # [0 0 1]
            affine_matrix[j, 0:2, 0:2] = np.frombuffer(
                f.read(16),
                dtype=np.float32,
                count=4,
            ).reshape(2, 2)

            # Get edges in counterclockwise direction.
            edges = (
                cell[index[j]][2:4].reshape(2, 1)
                / 2
                * np.array([[1, -1, -1, 1], [1, 1, -1, -1]])
            )

            # Transform edges.
            edges = np.matmul(affine_matrix[j, 0:2, 0:2], edges)
            edges.sort()

            # Get bottom right corner.
            b[..., j] = a[..., j] + edges[..., -1] - edges[..., 0] + np.array([2, 2])

            # Get alpha.
            if is_alpha == 1:
                alpha[j] = int.from_bytes(f.read(1), byteorder="little", signed=False)
            else:
                alpha[j] = 255

        # Get canvas dimensions.
        c = a.copy()
        c.sort()
        d = b.copy()
        d.sort()

        canvas_dim = np.array(d[..., -1] - c[..., 0], dtype=int)

        # Read frames info.
        frames_items = [np.zeros(0) for _ in range(blocks)]
        for i in range(blocks):
            subcells = int.from_bytes(f.read(2), byteorder="little", signed=False)
            f.seek(f.tell() + 1)  # extra byte to ignore.

            frames_items[i] = np.zeros(subcells, dtype=int)
            for j in range(subcells):
                frames_items[i][j] = int.from_bytes(
                    f.read(2), byteorder="little", signed=False
                )

        # Get states info.
        statenumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
        statenames = ["" for _ in range(statenumber)]
        stateitems = [0 for _ in range(statenumber)]

        for s in range(statenumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)

            statenames[s] = f.read(skip - 1).decode("utf8")
            f.seek(f.tell() + 1)

            start = int.from_bytes(f.read(2), byteorder="little", signed=False)
            end = int.from_bytes(f.read(2), byteorder="little", signed=False)
            stateitems[s] = end - start + 1

        # Generate the frames.
        def generate_frames(
            rgb_img,
            cell,
            blocks,
            index,
            a,
            c,
            affine_matrix,
            canvas_dim,
            frames_items,
            show_progress,
            prefix_str,
        ):
            for i in range(blocks):
                with Image(
                    width=canvas_dim[0], height=canvas_dim[1], background="transparent"
                ) as frame_img:
                    if show_progress is True:
                        report_progress(prefix_str, f"[{i + 1}/{blocks}]")

                    for j in reversed(frames_items[i]):
                        # Crop the cell.
                        with rgb_img.clone() as subcell:
                            subcell.crop(
                                cell[index[j]][0],
                                cell[index[j]][1],
                                width=cell[index[j]][2],
                                height=cell[index[j]][3],
                            )

                            subcell.image_set(subcell.fx(f"u * {alpha[j]}/255"))
                            subcell.virtual_pixel = "transparent"
                            subcell.distort(
                                "affine_projection",
                                affine_matrix[j, 0:3, 0:2].flatten().tolist(),
                                best_fit=True,
                            )

                            # Get coordinates.
                            coords = a[..., j] - c[..., 0]

                            # Set composition.
                            frame_img.composite(
                                image=subcell,
                                left=coords[0],
                                top=coords[1],
                            )
                    yield frame_img

        frame_iterator = generate_frames(
            rgb_img,
            cell,
            blocks,
            index,
            a,
            c,
            affine_matrix,
            canvas_dim,
            frames_items,
            show_progress,
            prefix_str,
        )

        return [frame_iterator, statenames, stateitems]
