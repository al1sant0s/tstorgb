from pathlib import Path
from wand.image import Image
from wand.drawing import Drawing
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="With this tool you can generate sprite sheets from previous images produced with the 'convert' tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "input_dir",
        help="List of comma separated directories containing the images to be used.",
    )

    parser.add_argument(
        "output_dir",
        help="Path to the directory where results will be stored.",
    )

    parser.add_argument(
        "--input_extension",
        help="The image format used for the imported images.",
        default="png",
    )

    parser.add_argument(
        "--output_extension",
        help="The image format used for the exported sprite sheet.",
        default="png",
    )

    parser.add_argument(
        "--image_quality",
        help="Percentage specifying image quality.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--border_color",
        help="The color of the borders between the frames on the sheet.",
        default="darksalmon",
    )

    parser.add_argument(
        "--label_background_color",
        help="The background color of the labels on the sheet.",
        default="indianred",
    )

    parser.add_argument(
        "--font",
        help="Font used for the labels on the sheet.",
        default="Alegreya ExtraBold, Arial-MT-Extra-Bold",
    )

    parser.add_argument(
        "--font_color",
        help="Font color used for the labels on the sheet.",
        default="antiquewhite",
    )

    args = parser.parse_args()

    out_path = Path(args.output_dir)

    if out_path.exists() is False:
        out_path.mkdir()

    # Get total of montages.
    n = 0
    total = np.sum(
        [len(list(Path(k).glob("*/"))) for k in [k for k in args.input_dir.split(",")]]
    )

    # Start making the montages.
    for in_path_str in args.input_dir.split(","):
        entity = Path(in_path_str)

        if entity.exists() is True:
            for variation in entity.glob("*/"):
                n += 1
                print(f"Montage: {n}/{total}", end="\r")
                with Image() as big_montage_img:
                    states = sorted([k.name for k in variation.glob("*/")])
                    if len(states) == 0:
                        max_frame_number = len(
                            list(variation.glob(f"*.{args.input_extension}"))
                        )

                        if max_frame_number == 0:
                            print(
                                f"No frames found for {variation}. Skipping this directory."
                            )
                            continue

                        with Image() as montage_img:
                            image_order = sorted(
                                [
                                    int(k.stem.rsplit("_image_", maxsplit=1)[-1])
                                    for k in variation.glob(f"*.{args.input_extension}")
                                ]
                            )
                            for index in image_order:
                                imagepath = Path(
                                    variation,
                                    f"{entity.name}_{variation.name}_image_{index}.{args.input_extension}",
                                )
                                with Image(filename=imagepath) as frame:
                                    frame.border(args.border_color, width=5, height=5)
                                    montage_img.image_add(frame)

                            montage_img.background_color = "transparent"
                            montage_img.compression_quality = args.image_quality
                            montage_img.montage(
                                thumbnail="+0+0",
                            )
                            montage_img.border(
                                color=args.border_color, width=5, height=5
                            )

                            big_montage_img.image_add(montage_img)

                        target = Path(out_path, entity.name)
                        target.mkdir(parents=True, exist_ok=True)

                        big_montage_img.background_color = "transparent"
                        big_montage_img.compression_quality = args.image_quality
                        big_montage_img.montage(tile="1x", thumbnail="+0+0")
                        big_montage_img.save(
                            filename=Path(
                                target,
                                f"{entity.name}_{variation.name}.{args.output_extension}",
                            )
                        )
                    else:
                        max_frame_number = max(
                            [
                                len(list(statename.glob(f"*.{args.input_extension}")))
                                for statename in variation.glob("*/")
                            ]
                        )

                        if max_frame_number == 0:
                            print(
                                f"No frames found for {variation}. Skipping this directory."
                            )
                            continue

                        nrow = int(np.sqrt(max_frame_number))
                        ncol = np.ceil(max_frame_number / nrow)
                        for statename in states:
                            statename = Path(variation, statename)
                            with Image() as montage_img:
                                for i in range(
                                    len(
                                        list(
                                            statename.glob(f"*.{args.input_extension}")
                                        )
                                    )
                                ):
                                    imagepath = Path(
                                        statename, f"{i}.{args.input_extension}"
                                    )

                                    with Image(filename=imagepath) as frame:
                                        frame.border(
                                            args.border_color, width=5, height=5
                                        )
                                        montage_img.image_add(frame)

                                montage_img.background_color = "transparent"
                                montage_img.compression_quality = args.image_quality
                                montage_img.montage(
                                    tile=f"{ncol}x",
                                    thumbnail="+0+0",
                                )
                                montage_img.border(
                                    color=args.border_color, width=5, height=5
                                )
                                # Write label.
                                montage_img.background_color = (
                                    args.label_background_color
                                )
                                montage_img.splice(
                                    x=0,
                                    y=0,
                                    width=0,
                                    height=256,
                                )
                                font_size = montage_img.width // len(statename.name)
                                with Drawing() as ctx:
                                    ctx.font_family = args.font
                                    ctx.font_style = "italic"
                                    ctx.font_size = min(11 / 8 * font_size, 200)
                                    ctx.text_kerning = 8
                                    ctx.fill_color = args.font_color
                                    montage_img.annotate(
                                        statename.name,
                                        ctx,
                                        left=64,
                                        baseline=190,
                                    )

                                big_montage_img.image_add(montage_img)

                        target = Path(out_path, variation.relative_to(entity.parent))
                        target.mkdir(parents=True, exist_ok=True)

                        big_montage_img.background_color = "transparent"
                        big_montage_img.compression_quality = args.image_quality
                        big_montage_img.montage(tile="1x", thumbnail="+0+0")
                        big_montage_img.save(
                            filename=Path(target, f"sheet.{args.output_extension}")
                        )

        else:
            print(
                f"Warning! The directory {in_path_str} does not exist. Skipping this directory."
            )
