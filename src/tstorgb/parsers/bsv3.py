import numpy as np
from pathlib import Path
from pyvips import Image
from tstorgb.parsers.rgb import rgb_parser
from tstorgb.parsers.addons.bsv3_addon import (
    crop_cells,
    get_states,
    frame_iterator,
)


def bsv3_parser(bsv3_file):
    with open(bsv3_file, "rb") as f:
        bsv3_set = set()

        check = int.from_bytes(f.read(2), byteorder="little", signed=False)

        if check in {259, 515}:
            # Read cell data.
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

            # Get rgb image.
            rgb_img_path = Path(bsv3_file.parent, bsv3_file.stem + ".rgb")
            if check == 515:
                skip = int.from_bytes(f.read(1), byteorder="little", signed=False)
                rgb_img_path = Path(
                    bsv3_file.parent, f.read(skip).decode("utf8")[0:-1].lower()
                )

            rgb_img = rgb_parser(rgb_img_path)
            bsv3_set.add(rgb_img_path.name)

            if rgb_img is False:
                return (None, list(), list(), set(), False)

            # Get cells.
            cells_imgs, cells_subregions, cells_names, bytepos = crop_cells(
                bsv3_file, bytepos=f.tell(), rgb_img=rgb_img, cellnumber=cellnumber
            )

            # Get frames.
            frames, statenames, stateitems = bsv3a(
                bsv3_file, bytepos, cells_imgs, cells_subregions, cells_names, is_alpha
            )

            return (frames, statenames, stateitems, bsv3_set, True)

        elif check in {771}:
            # Read cell data.
            unknown = np.frombuffer(f.read(4), dtype=np.float32)
            cellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)
            is_alpha = int.from_bytes(f.read(1), byteorder="little", signed=False)

            # Get rgb image.
            rgb_img_path = Path(bsv3_file.parent, bsv3_file.stem + ".rgb")
            rgb_img = rgb_parser(rgb_img_path)
            bsv3_set.add(rgb_img_path.name)

            if rgb_img is False:
                return (None, list(), list(), set(), False)

            # Get cells.
            cells_imgs, cells_subregions, cells_names, bytepos = crop_cells(
                bsv3_file,
                bytepos=f.tell(),
                rgb_img=rgb_img,
                cellnumber=cellnumber,
            )

            # Get frames.
            frames, statenames, stateitems = bsv3b(
                bsv3_file, bytepos, cells_imgs, cells_subregions, cells_names, is_alpha
            )

            return (frames, statenames, stateitems, bsv3_set, True)
        else:
            # Unsupported or invalid file.
            return (None, list(), list(), set(), False)


def bsv3a(bsv3_file, bytepos, cells_imgs, cells_subregions, cells_names, is_alpha):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)

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
                    tlc[i][0, j] = np.round(tlc[i][0, j]) - cells_subregions[index][2] + cells_subregions[index][0]

                tlc[i][1, j] = np.floor(tlc[i][1, j]) - cells_subregions[index][1]

                # Decide if multilayer processing should be done when rendering the frames.
                # Multilayer is very slow but it prevents alpha overlapping (brighter pixels in the intersections).
                # Only performs if there's at least one semitranslucent cropped subcell.
                if "_crop" in cells_names[index]:
                    if (
                        subcells_imgs[i][j][3].maxpos()[0] < 245
                        or 0
                        < cells_imgs[index]
                        .crop(
                            0.4 * cells_imgs[index].width,
                            0.4 * cells_imgs[index].height,
                            0.2 * cells_imgs[index].width + 1,
                            0.2 * cells_imgs[index].height + 1,
                        )[3]
                        .minpos()[0]
                        < 245
                    ):
                        subcells_layers[i].add(
                            cells_names[index].split("_crop", maxsplit=1)[0]
                        )
                    subcells_imgs[i][j] = subcells_imgs[i][j].affine(
                        affine_matrix[i][j, ...].flatten().tolist(), extend="copy"
                    )
                else:
                    subcells_imgs[i][j] = subcells_imgs[i][j].affine(
                        affine_matrix[i][j, ...].flatten().tolist(), extend="background"
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

        # Get states info.
        statenames, stateitems = get_states(bsv3_file, f.tell())

        # Get frames.
        frames = frame_iterator(
            canvas_img, subcells_imgs, subcells_names, subcells_layers, tlc
        )
        return (frames, statenames, stateitems)


def bsv3b(bsv3_file, bytepos, cells_imgs, cells_subregions, cells_names, is_alpha):
    with open(bsv3_file, "rb") as f:
        f.seek(bytepos)

        # Blocks information.
        blocks = int.from_bytes(f.read(2), byteorder="little", signed=False)
        subcellnumber = int.from_bytes(f.read(2), byteorder="little", signed=False)

        # Subcell data.
        subcells_imgs = [Image.black(1, 1) for _ in range(subcellnumber)]  # type: ignore
        subcells_names = ["" for _ in range(subcellnumber)]
        subcells_layers = set()
        tlc = np.zeros((2, subcellnumber), dtype=np.float32)
        brc = np.zeros((2, subcellnumber), dtype=np.float32)
        affine_matrix = np.array([np.identity(2)] * subcellnumber, dtype=np.float32)
        alpha = np.zeros(subcellnumber, dtype=int)

        # Process subcells.
        for j in range(subcellnumber):
            index = int.from_bytes(f.read(2), byteorder="little", signed=False)

            # Get top left corner.
            tlc[..., j] = np.frombuffer(
                f.read(8), dtype=np.float32, count=2
            ).transpose()

            # Get coefficients for the affine matrix.
            # [sx rx]
            # [ry sy]
            affine_matrix[j, ...] = (
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
                alpha[j] = int.from_bytes(f.read(1), byteorder="little", signed=False)
            else:
                alpha[j] = 255

            # Process subcells.
            subcells_names[j] = cells_names[index]
            subcells_imgs[j] = cells_imgs[index].copy()
            subcells_imgs[j] *= [1, 1, 1, alpha[j] / 255]

            # Adjust coordinates.
            if np.linalg.det(affine_matrix[j, ...]) > 0:
                tlc[0, j] = np.round(tlc[0, j]) - cells_subregions[index][0]
            else:
                tlc[0, j] = np.round(tlc[0, j]) - cells_subregions[index][2] + cells_subregions[index][0]

            tlc[1, j] = np.floor(tlc[1, j]) - cells_subregions[index][1]

            # Decide if multilayer processing should be done when rendering the frames.
            # Multilayer is very slow but it prevents alpha overlapping (brighter pixels in the intersections).
            # Only performs if there's at least one semitranslucent cropped subcell.
            if "_crop" in cells_names[index]:
                if (
                    subcells_imgs[j][3].maxpos()[0] < 245
                    or 0
                    < cells_imgs[index]
                    .crop(
                        0.4 * cells_imgs[index].width,
                        0.4 * cells_imgs[index].height,
                        0.2 * cells_imgs[index].width + 1,
                        0.2 * cells_imgs[index].height + 1,
                    )[3]
                    .minpos()[0]
                    < 245
                ):
                    subcells_layers.add(
                        cells_names[index].split("_crop", maxsplit=1)[0]
                    )
                subcells_imgs[j] = subcells_imgs[j].affine(
                    affine_matrix[j, ...].flatten().tolist(), extend="copy"
                )
            else:
                subcells_imgs[j] = subcells_imgs[j].affine(
                    affine_matrix[j, ...].flatten().tolist(), extend="background"
                )

            # Get bottom right corner.
            brc[..., j] = tlc[..., j] + np.array(
                [subcells_imgs[j].width, subcells_imgs[j].height]
            )

        # Get canvas dimensions.
        c = tlc.copy()
        c.sort()
        d = brc.copy()
        d.sort()

        canvas_dim = np.array(np.ceil(d[..., -1] - c[..., 0]), dtype=int)
        canvas_img = Image.new_from_array(
            np.zeros((canvas_dim[1], canvas_dim[0], 4)), interpretation="srgb"
        )

        # Correct coordinates.
        tlc -= np.array(c[..., 0]).reshape(2, 1)

        # Read frames info.
        frames_items = [list() for _ in range(blocks)]
        frames_items_names = [list() for _ in range(blocks)]
        frames_items_layers = [subcells_layers for _ in range(blocks)]
        frames_items_tlc = [np.array([]) for _ in range(blocks)]
        for i in range(blocks):
            subcells = int.from_bytes(f.read(2), byteorder="little", signed=False)
            f.seek(f.tell() + 1)  # extra byte to ignore.

            frames_items[i] = [Image.black(1, 1) for _ in range(subcells)]  # type: ignore
            frames_items_names[i] = ["" for _ in range(subcells)]
            frames_items_tlc[i] = np.zeros((2, subcells))
            for j in range(subcells):
                index = int.from_bytes(f.read(2), byteorder="little", signed=False)
                frames_items[i][j] = subcells_imgs[index].copy()
                frames_items_names[i][j] = subcells_names[index]
                frames_items_tlc[i][..., j] = tlc[..., index]

        # Get states info.
        statenames, stateitems = get_states(bsv3_file, f.tell())

        # Get frames.
        frames = frame_iterator(
            canvas_img,
            frames_items,
            frames_items_names,
            frames_items_layers,
            frames_items_tlc,
        )
        return (frames, statenames, stateitems)
