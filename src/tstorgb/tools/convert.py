import argparse
from pathlib import Path
from zipfile import ZipFile, is_zipfile
from pyvips import Image, GValue, cache_set_max
from tstorgb.parsers import bcell_parser, rgb_parser, bsv3_parser
from tstorgb.tools.progress import report_progress


def progress_str(n, total, filestem, extension):
    return f"Progress ({n * 100 / total:.2f}%) : [{total - n} file(s) left] ---> {filestem}.{extension}"


# Warning: this script requires libvips to work. If you do not have installed in your system,
# you will have to install it first before using the script.


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
        "--disable_bcell",
        help="If this option is enabled, bcell files will be ignored.",
        action="store_true",
    )

    parser.add_argument(
        "--disable_shadows",
        help="If this option is enabled, shadows from bcell files will be ignored.",
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
        You can choose between most of libvips supported image formats like png, webp, jpg, etc.
        Check libvips documentation for more info about the image formats you can use.
        """,
        default="png",
    )
    parser.add_argument(
        "--sequential",
        help="""
        Produce animated images for specific image formats (e.g. webp).
        """,
        action="store_true",
    )
    parser.add_argument(
        "--sequential_delay",
        help="""
        Time delay in miliseconds to wait between frames for the animated images produced with the --sequential argument.
        """,
        default=100,
        type=int,
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

    # Set libvips cache in order to lower memory usage.
    cache_set_max(0)

    # Help with the progress report.
    n = 0
    total = 0

    # keep track of bcell files.
    bcell_set = set("")

    # Keep track of bsv3 files.
    bsv3_set = set("")

    print("\n\n--- CONVERTING RGB FILES ---\n\n")

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
        + [len(list(Path(directory).glob("**/*.bsv3"))) for directory in directories]
    )

    if total == 0:
        print("[No file(s) found!]\n\n")
        print("Warning! No rgb/bsv3/bcell files found in the specified directories.")
        print(
            "Remember you should specify the directories where the files are, not the files themselves."
        )
        print("If you need help, execute the following command: tstorgb --help\n\n")
        return

    print(f"[{total} file(s) found!]\n\n")

    print("--- Starting conversion of rgb images ---")

    # Set destination of the converted rgb files.
    target = Path(args.output_dir)
    target.mkdir(exist_ok=True)

    for directory in directories:
        # Process bcell files.
        if args.disable_bcell is False:
            for bcell_file in directory.glob("**/*.bcell"):
                frames, framenumber, new_set, success = bcell_parser(
                    bcell_file, disable_shadows=args.disable_shadows
                )

                n += 1

                # Unsupported or invalid bcell file.
                if success is False:
                    # Clear line.
                    print(150 * " ", end="\r")
                    print(
                        f"Unknown bcell signature, invalid or missing rgb file. Skipping this file -> {bcell_file.name}!"
                    )
                    continue

                bcell_set = bcell_set.union(new_set)

                entity = bcell_file.stem.split("_", maxsplit=1)
                if len(entity) == 2:
                    target = Path(args.output_dir, entity[0], entity[1])
                else:
                    target = Path(args.output_dir, entity[0], "_default")

                target.mkdir(parents=True, exist_ok=True)

                # How the frames will be saved.
                if args.sequential is True:
                    report_frames = (
                        report_progress(
                            progress_str(n, total, bcell_file.stem, "bcell"),
                            f"[{i + 1}/{framenumber}]",
                        )
                        for i in range(framenumber)
                    )

                    animation_frames = (
                        Image.new_from_buffer(  # type: ignore
                            next(frames).write_to_buffer(  # type: ignore
                                f".{args.output_extension}",
                                Q=args.image_quality,
                            ),
                            options="",
                            access="sequential",
                        )
                        for _ in report_frames
                    )

                    animation = next(animation_frames).pagejoin(  # type: ignore
                        list(animation_frames)
                    )

                    animation.set_type(
                        GValue.array_int_type,
                        "delay",
                        [args.sequential_delay for _ in range(framenumber)],
                    )

                    animation.write_to_file(  # type: ignore
                        Path(
                            target,
                            f"{bcell_file.stem}.{args.output_extension}",
                        ),
                        Q=args.image_quality,
                    )

                else:
                    for i in range(framenumber):
                        next(frames).write_to_file(  # type: ignore
                            Path(
                                target,
                                f"{i}.{args.output_extension}",
                            ),
                            Q=args.image_quality,
                        )
                        report_progress(
                            progress_str(n, total, bcell_file.stem, "bcell"),
                            f"[{i + 1}/{framenumber}]",
                        )

        else:
            n += len(list(directory.glob("**/*.bcell")))

        # Process bsv3 files.
        if args.disable_bsv3 is False:
            for bsv3_file in directory.glob("**/*.bsv3"):
                frames, statenames, stateitems, new_set, success = bsv3_parser(
                    bsv3_file
                )

                framenumber = sum(stateitems)

                n += 1

                # Unsupported or invalid bsv3 file.
                if success is False:
                    # Clear line.
                    print(150 * " ", end="\r")
                    print(
                        f"Unknown bsv3 signature, invalid or missing rgb file. Skipping this file -> {bsv3_file.name}!"
                    )
                    continue

                bsv3_set = bsv3_set.union(new_set)

                entity = bsv3_file.stem.split("_", maxsplit=1)
                if len(entity) == 2:
                    target = Path(args.output_dir, entity[0], entity[1])
                else:
                    target = Path(args.output_dir, entity[0], "_default")

                target.mkdir(parents=True, exist_ok=True)

                # How the frames will be saved.
                for s, t, u in zip(statenames, stateitems, range(len(stateitems))):
                    dest = Path(target, s)
                    dest.mkdir(exist_ok=True)
                    if args.sequential is True:
                        report_frames = (
                            report_progress(
                                progress_str(n, total, bsv3_file.stem, "bsv3"),
                                f"[{i + 1 + sum(stateitems[:u])}/{framenumber}]",
                            )
                            for i in range(t)
                        )

                        animation_frames = (
                            Image.new_from_buffer(  # type: ignore
                                next(frames).write_to_buffer(  # type: ignore
                                    f".{args.output_extension}",
                                    Q=args.image_quality,
                                ),
                                options="",
                                access="sequential",
                            )
                            for _ in report_frames
                        )

                        animation = next(animation_frames).pagejoin(  # type: ignore
                            list(animation_frames)
                        )

                        animation.set_type(
                            GValue.array_int_type,
                            "delay",
                            [args.sequential_delay for _ in range(framenumber)],
                        )

                        animation.write_to_file(  # type: ignore
                            Path(
                                dest,
                                f"{s}.{args.output_extension}",
                            ),
                            Q=args.image_quality,
                        )

                    else:
                        for i in range(t):
                            next(frames).write_to_file(  # type: ignore
                                Path(
                                    dest,
                                    f"{i}.{args.output_extension}",
                                ),
                                Q=args.image_quality,
                            )
                            report_progress(
                                progress_str(n, total, bsv3_file.stem, "bsv3"),
                                f"[{i + 1 + sum(stateitems[:u])}/{framenumber}]",
                            )

        else:
            n += len(list(directory.glob("**/*.bsv3")))

        # Process the remaining rgb files.
        for rgb_file in directory.glob("**/*.rgb"):
            n += 1
            report_progress(progress_str(n, total, rgb_file.stem, "rgb"), "")

            # Image already processed.
            if rgb_file.name in bcell_set or rgb_file.name in bsv3_set:
                continue

            entity = rgb_file.stem.split("_", maxsplit=1)

            if len(entity) == 2:
                target = Path(
                    args.output_dir, entity[0], entity[1].split("_image", maxsplit=1)[0]
                )
            else:
                target = Path(args.output_dir, entity[0], "_default")

            target.mkdir(parents=True, exist_ok=True)

            rgb_image = rgb_parser(rgb_file)

            # Ignore this rgb_file if it cannot be parsed.
            if rgb_image is False:
                # Clear line.
                print(150 * " ", end="\r")
                print(
                    f"Invalid or missing rgb file. Skipping this file -> {rgb_file.name}!"
                )
                continue

            rgb_image.write_to_file(  # type: ignore
                Path(target, f"{rgb_file.stem}.{args.output_extension}"),
                Q=args.image_quality,
            )

    print("\n\n--- JOB COMPLETED!!! ---\n\n")
