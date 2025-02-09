import numpy as np
from pyvips import Image


def crop_cells(bsv3_file, bytepos, rgb_img, cellnumber):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        cells_imgs = [Image.black(1, 1) for _ in range(cellnumber)]  # type: ignore
        cells_subregions = [np.array([]) for _ in range(cellnumber)]
        cells_names = ["" for _ in range(cellnumber)]

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            cellname = f.read(skip).decode("utf8")[0:-1].lower()

            cells_names[i] = cellname

            # Read x, y, width and height.
            regions = np.frombuffer(f.read(8), dtype=np.uint16).reshape(4)

            x = int(regions[0])
            y = int(regions[1])
            w = int(regions[2])
            h = int(regions[3])

            # Check 4 edges.
            dx = 0
            dy = 0
            dw = 0
            dh = 0
            img_array = rgb_img.numpy()

            if x - 1 >= 0 and np.any(img_array[y : y + h, x - 1, :]) is np.True_:
                dx -= 1
                dw += 1
            if (
                x + w < img_array.shape[1]
                and np.any(img_array[y : y + h, x + w, :]) is np.True_
            ):
                dw += 1

            if y - 1 >= 0 and np.any(img_array[y - 1, x : x + w, :]) is np.True_:
                dy -= 1
                dh += 1

            if (
                y + h < img_array.shape[0]
                and np.any(img_array[y + h, x : x + w, :]) is np.True_
            ):
                dh += 1

            cells_imgs[i] = rgb_img.crop(x + dx, y + dy, w + dw, h + dh)
            cells_subregions[i] = np.array([-dx, -dy, dw, dh], dtype=int)

        return (cells_imgs, cells_subregions, cells_names, f.tell())


def get_frama_data(bsv3_file, bytepos, cells_imgs, cells_subregions, cells_names, is_alpha, blocks, check):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Subcell data.
        subcells_imgs = [list() for _ in range(blocks)]
        subcells_names = [list() for _ in range(blocks)]
        subcells_layers = [set() for _ in range(blocks)]
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
            subcells_names[i] = ["" for _ in range(subcellnumber)]
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

                # Get alpha.
                if is_alpha == 1:
                    alpha[i][j] = int.from_bytes(
                        f.read(1), byteorder="little", signed=False
                    )
                else:
                    alpha[i][j] = 255

                # Process subcells.
                subcells_names[i][j] = cells_names[index]
                subcells_imgs[i][j] = cells_imgs[index].copy()
                subcells_imgs[i][j] *= [1, 1, 1, alpha[i][j] / 255]

                # Adjust coordinates.
                if np.linalg.det(affine_matrix[i][j, ...]) > 0:
                    tlc[i][0, j] = np.round(tlc[i][0, j]) - cells_subregions[index][0]
                else:
                    tlc[i][0, j] = np.floor(tlc[i][0, j]) - cells_subregions[index][2] + cells_subregions[index][0]

                tlc[i][1, j] = np.floor(tlc[i][1, j]) - cells_subregions[index][1]

                # Decide if multilayer processing should be done when rendering the frames.
                # Multilayer is very slow but it prevents alpha overlapping (brighter pixels in the intersections).
                # Only performs if there's at least one semitranslucent cropped subcell.
                extend = "background"
                if "_crop" in cells_names[index]:

                    alpha_mask = subcells_imgs[i][j][3].maxpos()[0]
                    sub_alpha_mask = subcells_imgs[i][j][3].crop(
                        0.4 * subcells_imgs[i][j].width,
                        0.4 * subcells_imgs[i][j].height,
                        0.2 * subcells_imgs[i][j].width + 1,
                        0.2 * subcells_imgs[i][j].height + 1,
                    )
                    sub_alpha_mask = (sub_alpha_mask < 255).ifthenelse(sub_alpha_mask, 0).avg()

                    if (alpha_mask > 0 and alpha_mask < 255) or sub_alpha_mask > 50:
                        subcells_layers[i].add(
                            cells_names[index].split("_crop", maxsplit=1)[0]
                        )
                        extend = "copy"

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

        return (canvas_img, subcells_imgs, subcells_names, subcells_layers, tlc, f.tell())


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


def frame_iterator(canvas_img, subcells_imgs, subcells_names, subcells_layers, tlc):
    for i, subcells in zip(range(len(subcells_imgs)), subcells_imgs):
        if len(subcells) == 0:
            yield canvas_img
            continue

        group_layers = set()
        compose_imgs = []
        compose_x = []
        compose_y = []
        for j, subcell_img in zip(reversed(range(len(subcells))), reversed(subcells)):
            x = int(tlc[i][0, j])
            y = int(tlc[i][1, j])
            subcellname_split = subcells_names[i][j].split("_crop", maxsplit=1)
            subcellname = subcellname_split[0]
            if subcellname in subcells_layers[i]:
                if subcellname not in group_layers:
                    group_layers.add(subcellname)
                    compose_imgs.append(
                        canvas_img.copy().composite2(subcell_img, "over", x=x, y=y)
                    )
                    compose_x.append(0)
                    compose_y.append(0)
                else:
                    base_mask = compose_imgs[-1] > 0
                    overlay_mask = subcell_img > 0

                    compose_imgs[-1] = base_mask.composite(
                        [overlay_mask, compose_imgs[-1], subcell_img],
                        mode=["dest-out", "in", "over"],
                        x=[x, 0, x],
                        y=[y, 0, y],
                    )
            else:
                compose_imgs.append(subcell_img)
                compose_x.append(x)
                compose_y.append(y)

        yield canvas_img.composite(
            compose_imgs,
            mode="over",
            x=compose_x,
            y=compose_y,
            premultiplied=False,
        )
