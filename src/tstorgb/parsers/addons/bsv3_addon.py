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
            cells_subregions[i] = np.array([-dx, -dy, w, h], dtype=int)

        return (cells_imgs, cells_subregions, cells_names, f.tell())


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
