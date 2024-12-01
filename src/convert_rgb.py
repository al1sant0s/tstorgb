import numpy as np
import argparse
from wand.image import Image
from wand.drawing import Drawing
from tsto_rgb_bsv3_converter import rgb_parser, bsv3_parser
from pathlib import Path
from zipfile import ZipFile, is_zipfile


def progress_str(n, total, filename, extension):
    return f"Progress ({n * 100 / total:.2f}%) : [{total - n} rgb file(s) left] ---> {filename}.{extension}"


# Warning: this script requires ImageMagick to work. If you do not have installed in your machine,
# you will have to install it before using the script.

# Settings
individual_frames = True  # If set to True, individual frames will be made. If set to false, a montage will be made.

# Edit this if you want a different resulting image format.
# You can choose between most of ImageMagick supported image formats like png32, tiff, jpg, etc.
# Check ImageMagick documentation for more info about the image formats you can use.
# Raw image formats are not supported though because it requires you to specify the depth and image size
# of the image being processed beforehand.
extension = "webp"

parser = argparse.ArgumentParser()

parser.add_argument(
    "input_dir",
    help="List of directories containing the rgb files.",
)

parser.add_argument(
    "output_dir",
    help="Path to the directory where results will be stored.",
)

parser.add_argument(
    "--image_quality",
    help="Percentage specifying image quality (default 100).",
    default=100,
    type=int,
)

parser.add_argument(
    "--disable_bsv3",
    help="If this option is enabled, bsv3 files will be ignored and animated sprites will not be generated. \
Leave this option alone if you want to get the complete animations.",
    action="store_true",
)

parser.add_argument(
    "--make_sheet",
    help="If this option is enabled, frames will be stored in a single image. Otherwise, individual frames will be saved.",
    action="store_true",
)

parser.add_argument(
    "--search_zip",
    help="If enabled, zip files named '1' within specified directories will be extracted.",
    action="store_true",
)

args = parser.parse_args()
directiories = [Path(item) for item in args.input_dir.split(" ")]

print("\n\n--- CONVERTING RGB FILES ---\n\n")

print("Getting list of files to extract - ", end="")

print("[DONE!]\n\n")

# Help with the progress report.
n = 1
total = 0

print("Counting the number of files to convert (this might take a while) - ", end="")

# Get total of files to convert and extract zipped files.
if args.search_zip:
    for directory in directiories:
        for item in directory.glob("**/1"):
            if is_zipfile(item) is True:
                with ZipFile(item) as ZObject:
                    ZObject.extractall(path=Path(item.parent, "extracted"))
                    total += [".rgb" in str(i) for i in ZObject.infolist()].count(True)

total += len([item for item in Path(args.input_dir).glob("**/*.rgb")])

print(f"[{total} file(s) found!]\n\n")

print("--- Starting conversion of rgb images ---")

for directory in directiories:
    for file in directory.glob("**/*.rgb"):
        filename = file.stem

        # Set destination of the converted rgb files.
        target = Path(args.output_dir)
        target.mkdir(exist_ok=True)

        entity = filename.split("_", maxsplit=1)

        if len(entity) == 2:
            target = Path(target, entity[0], entity[1].split("_image", maxsplit=1)[0])
        else:
            target = Path(target, entity[0], "_default")

        target.mkdir(parents=True, exist_ok=True)

        rgb_image = rgb_parser(file, True, progress_str(n, total, filename, "rgb"))

        # Ignore this file if it cannot be parsed.
        if rgb_image is False:
            n += 1
            continue

        with rgb_image as baseimage:
            bsv3_file = Path(file.parent, filename + ".bsv3")

            # Save image or process animation.
            if args.disable_bsv3 or bsv3_file.exists() is False:
                baseimage.save(filename=Path(target, f"{filename}.{extension}"))

            else:
                bsv3_result = bsv3_parser(
                    bsv3_file, baseimage, True, progress_str(n, total, filename, "bsv3")
                )

                if bsv3_result is False:
                    baseimage.save(filename=Path(target, f"{filename}.{extension}"))
                    continue

                # How the frames will be saved.
                if args.make_sheet:
                    max_number_frames = max(bsv3_result[2])
                    montage_max_nrow = int(np.sqrt(max_number_frames))
                    montage_max_ncol = int(
                        np.ceil(max_number_frames / montage_max_nrow)
                    )

                    with Image() as big_montage:
                        for s, t in zip(bsv3_result[1], bsv3_result[2]):
                            with Image() as montage_img:
                                for i in range(t):
                                    with next(bsv3_result[0]) as frame_img:
                                        frame_img.border(
                                            color="darksalmon",
                                            width=5,
                                            height=5,
                                        )
                                        montage_img.image_add(frame_img)

                                montage_img.background_color = "transparent"
                                montage_img.compression_quality = args.image_quality
                                montage_img.montage(
                                    tile=f"{montage_max_ncol}x",
                                    thumbnail="+0+0",
                                )
                                montage_img.border(
                                    color="darksalmon", width=5, height=5
                                )
                                # Write state label.
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
                                        s,
                                        ctx,
                                        left=64,
                                        baseline=190,
                                    )

                                    big_montage.image_add(montage_img)

                        big_montage.background_color = "transparent"
                        big_montage.compression_quality = args.image_quality
                        big_montage.montage(tile="1x", thumbnail="+0+0")

                        # Save the final result.
                        finalresult = Path(target, f"{filename}.{extension}")
                        big_montage.save(filename=finalresult)

                else:
                    for s, t in zip(bsv3_result[1], bsv3_result[2]):
                        dest = Path(target, s)
                        dest.mkdir(exist_ok=True)
                        for i in range(t):
                            with next(bsv3_result[0]) as frame_img:
                                frame_img.compression_quality = args.image_quality
                                frame_img.save(
                                    filename=Path(
                                        dest,
                                        f"{i}.{extension}",
                                    )
                                )

        n += 1

print("\n\n--- JOB COMPLETED!!! ---\n\n")
