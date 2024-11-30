from contextlib import ExitStack
from genericpath import exists
from os import mkdir
from pathlib import Path
from numpy import var
from wand.image import Image
from wand.drawing import Drawing
import math
import argparse

input_extension = "webp"
output_extension = "png"

parser = argparse.ArgumentParser()
parser.add_argument(
    "input",
    help="list of directories containing the images for the montage",
)
parser.add_argument(
    "output",
    help="path to the directory where results will be stored",
)

args = parser.parse_args()

out_path = Path(args.output)

if out_path.exists() is False:
    out_path.mkdir()

for in_path_str in args.input.split(" "):
    entity = Path(in_path_str)

    if entity.exists() is True:
        total = len(list(entity.glob("*/")))
        for variation, m in zip(entity.glob("*/"), range(total)):
            print(f"Montage: {m + 1}/{total}")
            with Image() as big_montage_img:
                max_frame_number = max(
                    [
                        len(list(statename.glob(f"*.{input_extension}")))
                        for statename in variation.glob("*/")
                    ]
                )

                nrow = int(math.sqrt(max_frame_number))
                ncol = math.ceil(max_frame_number / nrow)
                states = sorted([k.name for k in variation.glob("*/")])
                if len(states) == 0:
                    with Image() as montage_img:
                        for i in range(
                            len(list(variation.glob(f"*.{input_extension}")))
                        ):
                            imagepath = Path(variation, f"{i}.{input_extension}")
                            with Image(filename=imagepath) as frame:
                                frame.border("darksalmon", width=5, height=5)
                                montage_img.image_add(frame)

                        montage_img.background_color = "transparent"
                        montage_img.montage(tile=f"{ncol}x", thumbnail="+0+0")
                        montage_img.border("darksalmon", width=5, height=5)
                        montage_img.background_color = "indianred"
                        montage_img.splice(
                            x=0,
                            y=0,
                            width=0,
                            height=256,
                        )
                        with Drawing() as ctx:
                            ctx.font_family = "Alegreya ExtraBold"
                            ctx.font_style = "italic"
                            ctx.font_size = 200
                            ctx.text_kerning = 8
                            ctx.fill_color = "antiquewhite"
                            montage_img.annotate(
                                variation.name,
                                ctx,
                                left=64,
                                baseline=190,
                            )
                        big_montage_img.image_add(montage_img)
                else:
                    for statename in states:
                        statename = Path(variation, statename)
                        with Image() as montage_img:
                            for i in range(
                                len(list(statename.glob(f"*.{input_extension}")))
                            ):
                                imagepath = Path(statename, f"{i}.{input_extension}")
                                with Image(filename=imagepath) as frame:
                                    frame.border("darksalmon", width=5, height=5)
                                    montage_img.image_add(frame)

                            montage_img.background_color = "transparent"
                            montage_img.montage(tile=f"{ncol}x", thumbnail="+0+0")
                            montage_img.border("darksalmon", width=5, height=5)
                            montage_img.background_color = "indianred"
                            montage_img.splice(
                                x=0,
                                y=0,
                                width=0,
                                height=256,
                            )
                            font_size = montage_img.width // len(statename.stem)
                            with Drawing() as ctx:
                                ctx.font_family = "Alegreya ExtraBold"
                                ctx.font_style = "italic"
                                ctx.font_size = min(11 / 8 * font_size, 200)
                                ctx.text_kerning = 8
                                ctx.fill_color = "antiquewhite"
                                montage_img.annotate(
                                    statename.stem,
                                    ctx,
                                    left=64,
                                    baseline=190,
                                )
                            big_montage_img.image_add(montage_img)
                big_montage_img.background_color = "transparent"
                big_montage_img.montage(tile="1x", thumbnail="+0+0")
                target = Path(out_path, variation.relative_to(entity.parent))
                target.mkdir(parents=True, exist_ok=True)
                big_montage_img.compression_quality = 100
                big_montage_img.save(filename=Path(target, f"sheet.{output_extension}"))

    else:
        print(
            f"Warning! The directory {in_path_str} does not exist. Skipping this directory."
        )
