import numpy as np
from wand.image import Image
from wand.drawing import Drawing
from tsto_rgb_bsv3_converter import rgb_parser, bsv3_parser
from pathlib import Path
from zipfile import ZipFile

# Warning: this script requires ImageMagick to work. If you do not have installed in your machine,
# you will have to install it before using the script.

# Settings
zipfilename = "1"
convertdir = "processed"
max_image_width = 16384
max_image_height = 16384
image_quality = 100
expand_bsv3 = True  # If set to True, bsv3 files will be processed (leave this value set to 1 to get the animated sprites).
individual_frames = False  # If set to True, individual frames will be made. If set to false, a montage will be made.

# Edit this if you want a different resulting image format.
# You can choose between most of ImageMagick supported image formats like png32, tiff, jpg, etc.
# Check ImageMagick documentation for more info about the image formats you can use.
# Raw image formats are not supported though because it requires you to specify the depth and image size
# of the image being processed beforehand.
extension = "jxl"

print("\n\n--- CONVERTING RGB FILES ---\n\n")

print("Getting list of files to extract - ", end="")

# Fetch a list of files to unzip.
ziplist = list(Path(Path.cwd()).glob(f"**/{zipfilename}"))

print("[DONE!]\n\n")

# Help with the progress report.
n = 1
total = 0


def progress_str(n, total, filename, extension):
    return f"Progress ({n * 100 / total:.2f}%) : [{total - n} rgb file(s) left] ---> {filename}.{extension}"


print("Counting the number of files to convert (this might take a while) - ", end="")

# Get total of files to convert.
for zipfile in ziplist:
    with ZipFile(zipfile) as zObject:
        total += [".rgb" in str(i) for i in zObject.infolist()].count(True)

print(f"[{total} file(s) found!]\n\n")

print("--- Starting conversion of rgb images ---")

for zipfile in ziplist:
    assets = Path(zipfile.parent, "assets")

    # Unzip file.
    with ZipFile(zipfile) as zObject:
        zObject.extractall(path=assets)

    for file in assets.glob("*.rgb"):
        filename = file.stem

        # Set destination of the converted rgb files.
        dest = Path(assets, convertdir)
        dest.mkdir(exist_ok=True)

        entity = filename.split("_")
        action = filename.split("_image")

        if len(action) > 1:
            target = Path(dest, entity[0], action[0])
        else:
            target = Path(dest, entity[0])

        target.mkdir(parents=True, exist_ok=True)

        rgb_image = rgb_parser(file, True, progress_str(n, total, filename, "rgb"))

        # Ignore this file if it cannot be parsed.
        if rgb_image is False:
            continue

        with rgb_image as baseimage:
            bsv3_file = Path(assets, filename + ".bsv3")

            # Save image or process animation.
            if expand_bsv3 is False or bsv3_file.exists() is False:
                baseimage.save(filename=Path(target, f"{filename}.{extension}"))
                n += 1

            else:
                frames = bsv3_parser(
                    bsv3_file, baseimage, True, progress_str(n, total, filename, "bsv3")
                )

                if frames is False:
                    baseimage.save(filename=Path(target, f"{filename}.{extension}"))
                    n += 1
                    continue

                with frames["frames"] as frames_img:
                    if individual_frames is True:
                        for s in frames["states"]:
                            for i in range(s["start"], s["end"] + 1):
                                frames_img.iterator_set(i)
                                with frames_img.image_get() as frame_img:
                                    frame_img.save(
                                        filename=Path(
                                            target,
                                            f"{s['statename']}_{i}.{extension}",
                                        )
                                    )

                    else:
                        with Image() as big_montage_img:
                            montage_height = np.zeros(len(frames["states"]), dtype=int)
                            for s, t in zip(
                                frames["states"], range(len(frames["states"]))
                            ):
                                with Image() as montage_img:
                                    for i in range(s["start"], s["end"] + 1):
                                        frames_img.iterator_set(i)
                                        with frames_img.image_get() as frame_img:
                                            frame_img.border(
                                                color="darksalmon", width=5, height=5
                                            )
                                            montage_img.image_add(frame_img)
                                    montage_img.background_color = "transparent"
                                    montage_img.montage(thumbnail="+0+0")
                                    montage_img.border(
                                        color="darksalmon", width=5, height=5
                                    )
                                    montage_height[t] = montage_img.height
                                    big_montage_img.image_add(montage_img)

                            big_montage_img.background_color = "transparent"
                            big_montage_img.montage(tile="1x", thumbnail="+0+0")
                            big_montage_img.background_color = "indianred"

                            # Write labels in the montage.
                            for s, t in zip(
                                frames["states"], range(len(frames["states"]))
                            ):
                                yoffset = 256 * t + montage_height[0:t].sum()
                                big_montage_img.splice(
                                    x=0,
                                    y=yoffset,
                                    width=0,
                                    height=256,
                                )
                                with Drawing() as ctx:
                                    ctx.font_family = "Alegreya ExtraBold"
                                    ctx.font_style = "italic"
                                    ctx.font_size = 200
                                    ctx.text_kerning = 8
                                    ctx.fill_color = "antiquewhite"
                                    big_montage_img.annotate(
                                        s["statename"],
                                        ctx,
                                        left=64,
                                        baseline=190 + yoffset,
                                    )

                            # Save the final result.
                            finalresult = Path(target, f"{filename}.{extension}")
                            big_montage_img.save(filename=finalresult)

print("\n\n--- JOB COMPLETED!!! ---\n\n")
