import numpy as np
from pyvips import Image


def rgb_parser(file, byte_seek=0):
    with open(file, "rb") as f:
        f.seek(byte_seek + 2)
        check = int.from_bytes(f.read(2), byteorder="little")

        width = int.from_bytes(f.read(2), byteorder="little")
        height = int.from_bytes(f.read(2), byteorder="little")

        pixel_data = np.zeros((height, width, 4))
        if check == 8192:
            for i in range(height):
                for j in range(width):
                    rgba_bits = int.from_bytes(
                        f.read(2), byteorder="little", signed=False
                    )
                    pixel_data[i, j, 0] = ((rgba_bits >> 12) & 15) * 255 / 15  # Red
                    pixel_data[i, j, 1] = ((rgba_bits >> 8) & 15) * 255 / 15  # Green
                    pixel_data[i, j, 2] = ((rgba_bits >> 4) & 15) * 255 / 15  # Blue
                    pixel_data[i, j, 3] = ((rgba_bits >> 0) & 15) * 255 / 15  # Alpha

            # Get base image
            return Image.new_from_array(pixel_data, interpretation="srgb").unpremultiply() # type: ignore
        elif check == 0:
            for i in range(height):
                for j in range(width):
                    pixel_data[i, j, 0] = int.from_bytes(f.read(1), signed=False)  # Red
                    pixel_data[i, j, 1] = int.from_bytes(
                        f.read(1), signed=False
                    )  # green
                    pixel_data[i, j, 2] = int.from_bytes(
                        f.read(1), signed=False
                    )  # Blue
                    pixel_data[i, j, 3] = int.from_bytes(
                        f.read(1), signed=False
                    )  # Alpha

            # Get base image
            return Image.new_from_array(pixel_data, interpretation="srgb") # type: ignore
        else:
            # Unsupported or invalid file.
            return False
