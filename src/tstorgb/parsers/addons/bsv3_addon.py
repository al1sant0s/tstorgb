import numpy as np
from pyvips import Image


def crop_cells(bsv3_file, bytepos, rgb_img, cellnumber):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        cells_imgs = [Image.black(1, 1) for _ in range(cellnumber)]  # type: ignore
        cells_subregions = [np.array([]) for _ in range(cellnumber)]

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            cellname = f.read(skip).decode("utf8")[0:-1].lower()

            # Read x, y, width and height.
            regions = np.frombuffer(f.read(8), dtype=np.uint16).reshape(4)

            x = int(regions[0])
            y = int(regions[1])
            w = max(1, int(regions[2]))
            h = max(1, int(regions[3]))

            # Check 4 edges.
            dx = 0
            dy = 0
            dw = 0
            dh = 0

            if x - 1 >= 0 and rgb_img[3].crop(x - 1, y, 1, h).maxpos()[0] > 0:
                dx -= 1
                dw += 1
            if x + w < rgb_img.width and rgb_img[3].crop(x + w, y, 1, h).maxpos()[0] > 0:
                dw += 1

            if y - 1 >= 0 and rgb_img[3].crop(x, y - 1, w, 1).maxpos()[0] > 0:
                dy -= 1
                dh += 1

            if y + h < rgb_img.height and rgb_img[3].crop(x, y + h, w, 1).maxpos()[0] > 0:
                dh += 1

            cells_imgs[i] = rgb_img.crop(x + dx, y + dy, w + dw, h + dh)
            cells_subregions[i] = np.array([-dx, -dy, dw, dh], dtype=int)

        return (cells_imgs, cells_subregions, f.tell())


def get_frama_data(bsv3_file, bytepos, cells_imgs, cells_subregions, is_alpha, blocks, check):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Subcell data.
        subcells_imgs = [list() for _ in range(blocks)]
        tlc = [np.array([]) for _ in range(blocks)]
        brc = [np.array([]) for _ in range(blocks)]
        affine_matrix = [np.array([]) for _ in range(blocks)]
        alpha = [np.array([]) for _ in range(blocks)]

        # Read blocks info.
        for i in range(blocks):
            subcellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Ignore extra byte.
            if check in {259, 515}:
                f.seek(f.tell() + 1)

            # Info about each subcell.
            subcells_imgs[i] = [
                Image.new_from_array(np.zeros((1, 1, 4)), interpretation="srgb")
                for _ in range(subcellnumber)
            ]
            tlc[i] = np.zeros((2, subcellnumber), dtype=np.float32)
            brc[i] = np.zeros((2, subcellnumber), dtype=np.float32)
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
                extend = "background" # Affine matrix default extend method.

                # Get alpha.
                if is_alpha == 1:
                    alpha[i][j] = int.from_bytes(
                        f.read(1), byteorder="little", signed=False
                    )
                else:
                    alpha[i][j] = 255

                # Process subcells.
                subcells_imgs[i][j] = cells_imgs[index].copy()
                subcells_imgs[i][j] *= [1, 1, 1, alpha[i][j] / 255]

                # Adjust coordinates.
                if np.linalg.det(affine_matrix[i][j, ...]) > 0:
                    tlc[i][0, j] = np.round(tlc[i][0, j]) - cells_subregions[index][0]
                else:
                    tlc[i][0, j] = np.floor(tlc[i][0, j]) - cells_subregions[index][2] + cells_subregions[index][0]

                tlc[i][1, j] = np.floor(tlc[i][1, j]) - cells_subregions[index][1]

                # Apply affine matrix transformation.
                subcells_imgs[i][j] = subcells_imgs[i][j].affine(
                    affine_matrix[i][j, ...].flatten().tolist(), extend=extend
                )

                # Get bottom right corner.
                brc[i][..., j] = tlc[i][..., j] + np.array(
                    [subcells_imgs[i][j].width, subcells_imgs[i][j].height]
                )

        # Standard canvas dimensions.
        canvas_dim = np.array([1, 1], dtype=int)

        # Make sure there is at least one subcell to process. Otherwise, the canvas will have 1x1 size.
        subcells_tlc = [tlc[i] for i in range(blocks) if len(subcells_imgs[i]) > 0]
        if len(subcells_tlc) > 0:
            # Get canvas dimensions.
            c = np.hstack(subcells_tlc)
            d = np.hstack([brc[i] for i in range(blocks) if len(subcells_imgs[i]) > 0])
            c.sort()
            d.sort()

            canvas_dim = np.array(np.ceil(d[..., -1] - c[..., 0]), dtype=int)

            # Correct coordinates.
            tlc = [tlc[i] - np.array(c[..., 0]).reshape(2, 1) for i in range(blocks)]

        canvas_img = Image.new_from_array(
            np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
        )

        return (canvas_img, subcells_imgs, tlc, f.tell())


def get_states(bsv3_file, bytepos):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

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

        return (statenames, stateitems)


def frame_iterator(canvas_img, subcells_imgs, tlc):
    for i, subcells in zip(range(len(subcells_imgs)), subcells_imgs):
        if len(subcells) == 0:
            yield canvas_img
            continue

        yield canvas_img.composite(
            list(reversed(subcells)),
            mode="over",
            x=list(reversed(tlc[i][0, ...])),
            y=list(reversed(tlc[i][1, ...])),
            premultiplied=False,
        )
