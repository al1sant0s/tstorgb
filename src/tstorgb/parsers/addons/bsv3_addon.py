import numpy as np
from pyvips import Image


def crop_cells(bsv3_file, bytepos, rgb_img, cellnumber):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        cells_imgs = [Image.black(1, 1) for _ in range(cellnumber)]  # type: ignore
        # cells_orientation = np.zeros((2, cellnumber))
        # img_array = rgb_img.numpy()

        # Get cells.
        for i in range(cellnumber):
            skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
            f.seek(f.tell() + skip)

            # Read x, y, width and height.
            region = np.frombuffer(f.read(8), dtype=np.uint16)

            x = int(region[0])
            y = int(region[1])
            w = int(region[2])
            h = int(region[3])

            # Check 4 edges.
            # dx = 0
            # dy = 0
            # if x - 1 >= 0 and np.any(img_array[y : y + h + 1, x - 1, :]) is np.True_:
            #    dx -= 1
            # if (
            #    x + w + 1 <= img_array.shape[1]
            #    and np.any(img_array[y : y + h + 1, x + w, :]) is np.True_
            # ):
            #    dx += 1

            # if y - 1 >= 0 and np.any(img_array[y - 1, x : x + w + 1, :]) is np.True_:
            #    dy -= 1

            # if (
            #    y + h + 1 < img_array.shape[0]
            #    and np.any(img_array[y + h, x : x + w + 1, :]) is np.True_
            # ):
            #    dy += 1

            # cells_orientation[..., i] = np.array([dx, dy], dtype=int)

            cells_imgs[i] = rgb_img.crop(x, y, w, h)

        return (cells_imgs, f.tell())


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
        )
