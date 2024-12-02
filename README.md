# Convert any RGB files from TSTO

This package allows you to convert the raw RGB assets from 'The Simpsons: Tapped Out' game into proper images.
It uses ImageMagick to perform the conversion. So you are required to install it in your system in order for the tool to work.
Multiple options are available for customizing the results. You can choose the file extension of the produced images, where to save it, etc.

# Installation

```python
python3 -m pip install @git+https://github.com/al1sant0s/tsto_rgb_bsv3_tools
```

# Simple usage

There are two builtin scripts in the src/ directory, the 'tsto_convert_rgb.py' will receive a list of comma separated directories to search for the rgb files and
it will save the results in a specified directory. For example, supposing the rgb files are inside a directory called "rgb_dir" and you want the images to be exported
to the "destination" directory, you would issue the following command:
```python
python3 src/tsto_convert_rgb.py rgb_dir destination
```

Check the help for more information.
```python
python3 src/tsto_convert_rgb.py --help
```
