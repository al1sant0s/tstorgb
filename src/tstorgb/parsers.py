import numpy as np
from pyvips import Image


def report_progress(prefix_str, parsing_info):
    # Clear line.
    print(150 * " ", end="\r")
    # Print progress.
    print(f"{prefix_str} {parsing_info}", end="\r")


def rgb_parser(file, show_progress=False, prefix_str=""):
    if show_progress is True:
        report_progress(prefix_str, "")

    with open(file, "rb") as f:
        f.seek(2)
        check = int.from_bytes(f.read(2), byteorder="little")

        width = int.from_bytes(f.read(2), byteorder="little")
        height = int.from_bytes(f.read(2), byteorder="little")

        pixel_data = np.zeros((height, width, 4))
        if check == 8192:
            for i in range(height):
                for j in range(width):
                    rgba_bits = int.from_bytes(
                        f.read(2), byteorder="little", signed=False
                    )
                    pixel_data[i, j, 0] = ((rgba_bits >> 12) & 15) * 255 / 15  # Red
                    pixel_data[i, j, 1] = ((rgba_bits >> 8) & 15) * 255 / 15  # Green
                    pixel_data[i, j, 2] = ((rgba_bits >> 4) & 15) * 255 / 15  # Blue
                    pixel_data[i, j, 3] = ((rgba_bits >> 0) & 15) * 255 / 15  # Alpha
        elif check == 0:
            for i in range(height):
                for j in range(width):
                    pixel_data[i, j, 0] = int.from_bytes(f.read(1), signed=False)  # Red
                    pixel_data[i, j, 1] = int.from_bytes(
                        f.read(1), signed=False
                    )  # green
                    pixel_data[i, j, 2] = int.from_bytes(
                        f.read(1), signed=False
                    )  # Blue
                    pixel_data[i, j, 3] = int.from_bytes(
                        f.read(1), signed=False
                    )  # Alpha
        else:
            # Unsupported signature,
            return False

        # Get base image
        return Image.new_from_array(pixel_data, interpretation="srgb").unpremultiply()  # type: ignore


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
            affine_matrix.append(np.array([np.identity(2)] * subcells[i]))  # type: ignore
            alpha.append(np.zeros(subcells[i], dtype=int))

            for j in range(subcells[i]):
                index[i][j] = int.from_bytes(
                    f.read(2), byteorder="little", signed=False
                )

                # Get top left corner.
                a[i][..., j] = np.frombuffer(
                    f.read(8), dtype=np.float32, count=2
                ).transpose()

                # Get coefficients for the affine matrix.
                # [sx rx]
                # [ry sy]
                affine_matrix[i][j, ...] = np.frombuffer(
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
                edges = np.matmul(affine_matrix[i][j, ...], edges)
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
                if show_progress is True:
                    report_progress(prefix_str, f"[{i + 1}/{blocks}]")

                frame_img = Image.new_from_array(
                    np.zeros((canvas_dim[1], canvas_dim[0], 4)),
                    interpretation="srgb",
                )

                if subcells[i] == 0:
                    yield frame_img
                    continue

                subcell_img = [0 for _ in range(subcells[i])]
                for j in range(subcells[i]):
                    # Crop the cell.
                    subcell_img[j] = rgb_img.crop(
                        cell[index[i][j]][0],
                        cell[index[i][j]][1],
                        cell[index[i][j]][2],
                        cell[index[i][j]][3],
                    )
                    subcell_img[j] *= [1, 1, 1, alpha[i][j] / 255]  # type: ignore
                    subcell_img[j] = subcell_img[j].affine(  # type: ignore
                        affine_matrix[i][j, ...].transpose().flatten().tolist(),
                        extend="background",
                    )

                yield frame_img.composite(  # type: ignore
                    list(reversed(subcell_img)),
                    mode="over",
                    x=list(
                        reversed(a[i][0, ...] - c[0, 0] + 1)
                    ),  # Adding one is necessary to centralize the sprite.
                    y=list(
                        reversed(a[i][1, ...] - c[1, 0] + 1)
                    ),  # Adding one is necessary to centralize the sprite.
                )

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
        affine_matrix = np.array([np.identity(2)] * subcells)
        alpha = np.zeros(subcells, dtype=int)

        for j in range(subcells):
            index[j] = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Get top left corner.
            a[..., j] = np.frombuffer(f.read(8), dtype=np.float32, count=2).transpose()

            # Get coefficients for the affine matrix.
            # [sx rx]
            # [ry sy]
            affine_matrix[j, ...] = np.frombuffer(
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
            edges = np.matmul(affine_matrix[j, ...], edges)
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
                if show_progress is True:
                    report_progress(prefix_str, f"[{i + 1}/{blocks}]")

                frame_img = Image.new_from_array(
                    np.zeros((canvas_dim[1], canvas_dim[0], 4)),
                    interpretation="srgb",
                )

                if frames_items[i].ndim == 0:
                    yield frame_img
                    continue

                subcell_img = [0 for _ in frames_items[i]]
                for j, k in zip(frames_items[i], range(len(subcell_img))):
                    # Crop the cell.
                    subcell_img[k] = rgb_img.crop(
                        cell[index[j]][0],
                        cell[index[j]][1],
                        cell[index[j]][2],
                        cell[index[j]][3],
                    )
                    subcell_img[k] *= [1, 1, 1, alpha[j] / 255]  # type: ignore
                    subcell_img[k] = subcell_img[k].affine(  # type: ignore
                        affine_matrix[j, ...].transpose().flatten().tolist(),
                        extend="background",
                    )

                yield frame_img.composite(  # type: ignore
                    list(reversed(subcell_img)),
                    mode="over",
                    x=list(
                        reversed(a[0, frames_items[i]] - c[0, 0] + 1)
                    ),  # Adding one is necessary to centralize the sprite.
                    y=list(
                        reversed(a[1, frames_items[i]] - c[1, 0] + 1)
                    ),  # Adding one is necessary to centralize the sprite.
                )

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
