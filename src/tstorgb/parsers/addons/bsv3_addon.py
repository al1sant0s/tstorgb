import numpy as np
from pyvips import Image


def crop_cells(bsv3_file, bytepos, rgb_img, cellnumber):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        cells_imgs = [Image.black(1, 1) for _ in range(cellnumber)]  # type: ignore
        cells_types = ["normal" for _ in range(cellnumber)]

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            cellname = f.read(skip).decode("utf8")[0:-1].lower()

            # Determine cell type for later operations.
            if "crop" in cellname:
                cells_types[i] = "crop"

            # Read x, y, width and height.
            region = np.frombuffer(f.read(8), dtype=np.uint16)

            x = int(region[0])
            y = int(region[1])
            w = int(region[2])
            h = int(region[3])

            cells_imgs[i] = rgb_img.crop(x, y, w, h)

        return (cells_imgs, cells_types, f.tell())


def check_cell_edges(cell_img):
    # Check 4 edges.

    x = 0
    y = 0
    w = cell_img.width
    h = cell_img.height
    img_array = cell_img.numpy()

    if np.any(img_array[y : y + h, x, 3]) is np.True_:
        return True

    elif np.any(img_array[y : y + h, x + w - 1, 3]) is np.True_:
        return True

    elif np.any(img_array[y, x : x + w, 3]) is np.True_:
        return True

    elif np.any(img_array[y + h - 1, x : x + w, 3]) is np.True_:
        return True

    else:
        return False


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


def frame_iterator(canvas_img, subcells_imgs, a):
    for i, subcells in zip(range(len(subcells_imgs)), subcells_imgs):
        if len(subcells) == 0:
            yield canvas_img
            continue

        yield canvas_img.composite(  # type: ignore
            list(reversed(subcells)),
            mode="over",
            x=list(reversed(np.array(a[i][0, ...], dtype=int))),
            y=list(reversed(np.array(a[i][1, ...], dtype=int))),
            premultiplied=False,
        )
