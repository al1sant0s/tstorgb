import numpy as np
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
            # Swap collor channels.
            rgb_img.image_add(img.channel_images["blue"])  # type: ignore
            rgb_img.image_add(img.channel_images["alpha"])  # type: ignore
            rgb_img.image_add(img.channel_images["red"])  # type: ignore
            rgb_img.image_add(img.channel_images["green"])  # type: ignore
            rgb_img.combine(colorspace="srgb")

        # Revert alpha premultiply.
        rgb_img.image_set(rgb_img.fx("u/u.a", channel="rgb"))

        return rgb_img.clone()


def bsv3_parser(bsv3_file, rgb_img, show_progress=False, prefix_str=""):
    with open(bsv3_file, "rb") as f:
        check = int.from_bytes(f.read(2), byteorder="little", signed=False)

        if check == 259:
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)
        elif check == 771:
            is_alpha = int(np.frombuffer(f.read(4), dtype=np.float32))
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            f.seek(f.tell() + 1)
        else:
            print("Invalid bsv3 signature. Skipping this file.")
            return False
            # rgb_img.save(filename=Path(target, f"{file.stem}.{extension}"))
            # n += 1

        cell = np.array([np.zeros(4)] * cellnumber, dtype=int)

        # Get cells.
        with Image() as cell_img:
            for i in range(cellnumber):
                skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
                f.seek(f.tell() + skip)

                # Read x, y, width and height.
                cell[i] = np.frombuffer(f.read(8), dtype=np.uint16)

                # Crop the cell.
                with rgb_img.clone() as img:
                    img.crop(
                        cell[i][0],
                        cell[i][1],
                        width=cell[i][2],
                        height=cell[i][3],
                    )
                    cell_img.image_add(img)

            # Frames information.
            frames = int.from_bytes(f.read(2), byteorder="little", signed=False)
            subcells = list()  # Number of subcells.

            # Subcell data.
            index = list()
            a = list()
            b = list()
            affine_matrix = list()
            alpha = list()

            # Read frames info.
            for i in range(frames):
                if show_progress is True:
                    report_progress(prefix_str, f"[{0}/{i + 1}]")

                subcells.append(
                    int.from_bytes(f.read(2), byteorder="little", signed=False)
                )

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
            states = [dict.fromkeys(["statename", "start", "end"])] * statenumber

            # Capture the frames.
            with Image() as frames_img:
                for s in range(statenumber):
                    skip = int.from_bytes(f.read(1), byteorder="little", signed=False)

                    statename = f.read(skip - 1).decode("utf8")
                    f.seek(f.tell() + 1)

                    start = int.from_bytes(f.read(2), byteorder="little", signed=False)
                    end = int.from_bytes(f.read(2), byteorder="little", signed=False)

                    with Image() as frame_img:
                        frame_img.background_color = "transparent"
                        # Generate the frames.
                        for i in range(start, end + 1):
                            if show_progress is True:
                                report_progress(prefix_str, f"[{i + 1}/{frames}]")

                            frame_img.image_add(
                                Image(
                                    width=canvas_dim[0],
                                    height=canvas_dim[1],
                                    background="transparent",
                                )
                            )

                            for j in reversed(range(subcells[i])):
                                cell_img.iterator_set(index[i][j])
                                with cell_img.image_get() as subcell:  # type: ignore
                                    subcell.image_set(
                                        subcell.fx(f"u * {alpha[i][j]}/255")
                                    )
                                    subcell.virtual_pixel = "transparent"
                                    subcell.distort(
                                        "affine_projection",
                                        affine_matrix[i][j, 0:3, 0:2]
                                        .flatten()
                                        .tolist(),
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

                        # Add to frames.
                        frames_img.image_add(frame_img)

                    # Register state.
                    states[s]["statename"] = statename
                    states[s]["start"] = start
                    states[s]["end"] = end

                return {"frames": frames_img.clone(), "states": states}
