import numpy as np
from pyvips import Image


def rgb_parser(file, byte_seek=0):
    if file.exists() is True:
        with open(file, "rb") as f:
            f.seek(byte_seek + 2)
            check = int.from_bytes(f.read(2), byteorder="little")

            width = int.from_bytes(f.read(2), byteorder="little")
            height = int.from_bytes(f.read(2), byteorder="little")

            pixel_data = np.zeros((height, width, 4))
            if check == 8192:
                buffer = np.frombuffer(
                    f.read(), dtype=np.uint16, count=width * height
                ).reshape(height, width)
                pixel_data[..., 0] = (
                    np.bitwise_and(np.right_shift(buffer, 12), 15) * 255 / 15
                )  # Red
                pixel_data[..., 1] = (
                    np.bitwise_and(np.right_shift(buffer, 8), 15) * 255 / 15
                )  # Green
                pixel_data[..., 2] = (
                    np.bitwise_and(np.right_shift(buffer, 4), 15) * 255 / 15
                )  # Blue
                pixel_data[..., 3] = (
                    np.bitwise_and(np.right_shift(buffer, 0), 15) * 255 / 15
                )  # Alpha

                # Get base image
                return Image.new_from_array(
                    pixel_data, interpretation="srgb"
                ).unpremultiply()


            elif check == 24576:
                buffer = np.frombuffer(
                    f.read(), dtype=np.uint16, count=width * height
                ).reshape(height, width)
                pixel_data[..., 0] = np.bitwise_and(
                    np.right_shift(buffer, 0), 255
                )  # Red
                pixel_data[..., 1] = np.bitwise_and(
                    np.right_shift(buffer, 0), 255
                )  # Green
                pixel_data[..., 2] = np.bitwise_and(
                    np.right_shift(buffer, 0), 255
                )  # Blue
                pixel_data[..., 3] = np.bitwise_and(
                    np.right_shift(buffer, 8), 255
                )  # Alpha

                # Get base image
                return Image.new_from_array(
                    pixel_data, interpretation="srgb"
                )

            elif check == 0:
                buffer = np.frombuffer(
                    f.read(), dtype=np.uint32, count=width * height
                ).reshape(height, width)
                pixel_data[..., 0] = np.bitwise_and(
                    np.right_shift(buffer, 0), 255
                )  # Red
                pixel_data[..., 1] = np.bitwise_and(
                    np.right_shift(buffer, 8), 255
                )  # Green
                pixel_data[..., 2] = np.bitwise_and(
                    np.right_shift(buffer, 16), 255
                )  # Blue
                pixel_data[..., 3] = np.bitwise_and(
                    np.right_shift(buffer, 24), 255
                )  # Alpha

                # Get base image
                return Image.new_from_array(pixel_data, interpretation="srgb")  # type: ignore

    # Unsupported or invalid file.
    return False
