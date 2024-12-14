import argparse
from tstorgb.parsers import bcell_parser, rgb_parser, bsv3_parser, report_progress
from pathlib import Path
from zipfile import ZipFile, is_zipfile


def progress_str(n, total, filestem, extension):
    return f"Progress ({n * 100 / total:.2f}%) : [{total - n} rgb file(s) left] ---> {filestem}.{extension}"


# Warning: this script requires libvips to work. If you do not have installed in your machine,
# you will have to install it before using the script.


def main():
    parser = argparse.ArgumentParser(
        description="""
        This tool allows you to convert the raw RGB assets from 'The Simpsons: Tapped Out' game into proper images.
        It uses libvips to perform the conversions, so make sure you have it installed in your system.
        Multiple options are available for customizing the results. You can choose the file extension of the produced images, where to save it, etc.
        Check the help for more information.
        """,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--search_zip",
        help="If enabled, zip files named '1' within specified directories will be extracted.",
        action="store_true",
    )

    parser.add_argument(
        "--disable_bsv3",
        help="If this option is enabled, bsv3 files will be ignored and animated sprites will not be generated. \
    Leave this option alone if you want to get the complete animations.",
        action="store_true",
    )

    parser.add_argument(
        "--image_quality",
        help="Percentage specifying image quality.",
        default=100,
        type=int,
    )

    parser.add_argument(
        "--output_extension",
        help="""
        Image format used for the exported images.
        You can choose between most of libvips supported image formats like png, webp, tiff, jpg, etc.
        Check libvips documentation for more info about the image formats you can use.
        """,
        default="png",
    )

    parser.add_argument(
        "input_dir",
        help="List of directories containing the rgb files.",
        nargs="+",
    )

    parser.add_argument(
        "output_dir",
        help="Directory where results will be stored.",
    )

    args = parser.parse_args()
    directories = [Path(item) for item in args.input_dir]

    # Help with the progress report.
    n = 1
    total = 0

    # keep track of bcell files.
    bcell_set = set()

    print("\n\n--- CONVERTING RGB FILES ---\n\n")

    print("Getting list of files to extract - ", end="")

    print("[DONE!]\n\n")

    print(
        "Counting the number of files to convert (this might take a while) - ", end=""
    )

    # Get total of files to convert and extract zipped files.
    if args.search_zip:
        for directory in directories:
            for item in directory.glob("**/1"):
                if is_zipfile(item) is True:
                    with ZipFile(item) as ZObject:
                        ZObject.extractall(path=Path(item.parent, "extracted"))

    total = sum(
        [len(list(Path(directory).glob("**/*.rgb"))) for directory in directories]
        + [len(list(Path(directory).glob("**/*.bcell"))) for directory in directories]
    )

    # if total == 0:
    #    raise Exception("No rgb files found in the specified directories.")

    print(f"[{total} file(s) found!]\n\n")

    print("--- Starting conversion of rgb images ---")

    # Set destination of the converted rgb files.
    target = Path(args.output_dir)
    target.mkdir(exist_ok=True)

    for directory in directories:
        for bcell_file in directory.glob("**/*.bcell"):
            frames, new_set, blocks = bcell_parser(bcell_file)

            bcell_set = bcell_set.union(new_set)

            entity = bcell_file.stem.split("_", maxsplit=1)
            if len(entity) == 2:
                target = Path(args.output_dir, entity[0], entity[1])
            else:
                target = Path(args.output_dir, entity[0], "_default")

            target.mkdir(parents=True, exist_ok=True)

            # How the frames will be saved.
            current_set = set()
            for i in range(blocks):
                frame_img, frame_name = next(frames)  # type: ignore
                frame_img.write_to_file(  # type: ignore
                    Path(
                        target,
                        f"{i}.{args.output_extension}",
                    ),
                    Q=args.image_quality,
                )
                report_progress(
                    progress_str(n, total, bcell_file.stem, "bcell"),
                    f"[{i + 1}/{blocks}]",
                )

                if frame_name not in current_set:
                    current_set.add(frame_name)
                    n += 1

        for file in directory.glob("**/*.rgb"):
            if file.name in bcell_set:
                continue

            entity = file.stem.split("_", maxsplit=1)

            if len(entity) == 2:
                target = Path(
                    args.output_dir, entity[0], entity[1].split("_image", maxsplit=1)[0]
                )
            else:
                target = Path(args.output_dir, entity[0], "_default")

            target.mkdir(parents=True, exist_ok=True)

            rgb_image = rgb_parser(file)
            report_progress(progress_str(n, total, file.stem, "rgb"), "")

            # Ignore this file if it cannot be parsed.
            if rgb_image is False:
                n += 1
                continue

            bsv3_file = Path(file.parent, file.stem + ".bsv3")

            # Save image or process animation.
            if args.disable_bsv3 or bsv3_file.exists() is False:
                rgb_image.write_to_file(  # type: ignore
                    Path(target, f"{file.stem}.{args.output_extension}"),
                    Q=args.image_quality,
                )

            else:
                frames, statenames, stateitems, blocks = bsv3_parser(
                    bsv3_file,
                    rgb_image,
                )

                if frames is False:
                    rgb_image.write_to_file(  # type: ignore
                        Path(target, f"{file.stem}.{args.output_extension}"),
                        Q=args.image_quality,
                    )
                    continue

                # How the frames will be saved.
                for s, t, u in zip(statenames, stateitems, range(len(stateitems))):
                    dest = Path(target, s)
                    dest.mkdir(exist_ok=True)
                    for i in range(t):
                        next(frames).write_to_file(  # type: ignore
                            Path(
                                dest,
                                f"{i}.{args.output_extension}",
                            ),
                            Q=args.image_quality,
                        )
                        report_progress(
                            progress_str(n, total, file.stem, "bsv3"),
                            f"[{i + 1 + sum(stateitems[:u])}/{blocks}]",
                        )

            n += 1

    print("\n\n--- JOB COMPLETED!!! ---\n\n")
