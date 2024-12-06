# Convert any RGB files from TSTO

This package allows you to convert the raw RGB assets from 'The Simpsons: Tapped Out' game into proper images.
It uses [**libvips**](https://www.libvips.org/) to perform the conversions. So you are required to install it before using the tool. Multiple options are available for customizing the results. You can choose the file extension of the produced images, where to save it, etc.

## Installation

```python
python3 -m pip install tstorgb@git+https://github.com/al1sant0s/tstorgb
```

## Simple usage

The convert tool will receive a list of comma separated directories to search for the rgb files, convert them into images of specificied format and then
it will save the results in a directory you provide. For example, supposing the rgb files are inside a directory called 'rgb_dir' and you want the images to be exported
to the 'destination' directory, you would issue the following command:

```
python3 -m tstorgb.convert rgb_dir destination
```

The images will be saved in subdirectories within the destination directory. Each subdirectory corresponds to a certain entity. For example, if there were two rgb files in 'rgb_dir', one named
'yellowhouse.rgb' and the other named 'orangehouse.rgb', the destination directory would have the following structure after the conversion:

- destination/
  - destination/yellowhouse/
  - destination/orangehouse/

Sometimes an entity also has variations. Suppose for example there exists 'yellowhouse_normal.rgb' and 'yellowhouse_premium.rgb'. Then this would be the resulting directory structure:

- destination/
  - destination/yellowhouse
    - destination/yellowhouse/normal
    - destination/yellowhouse/premium
  - destination/orangehouse
    - destination/orangehouse/_default

In theory, the entity is just the name that preecedes the first underscore character in a filename and the variation is the rest after the underscore excluding the .rgb extension.
For example, given the name 'something_anything_else.rgb', something is the entity and anything_else is the variation.
When a file corresponding to an entity doesn't specifiy a variation (a filename without an underscore in it like 'orangehouse.rgb') the _default variation subdirectory will be created.

## Main arguments

If you have your rgb files inside a zip file named '1', you can pass the --search_zip argument to deal with the extraction.

```
python3 -m tstorgb.convert --search_zip rgb_dir destination
```

If you prefer to use another file extension than png you can use the --output_extension argument.

```
python3 -m tstorgb.convert --search_zip --output_extension webp rgb_dir destination
```

For more information about the other arguments the 'convert' tool supports check the help.

```
python3 -m tstorgb.convert --help
```

## Disable bsv3 processing

The bsv3 files are used to generate the correct frames. If for some reason you don't want to process the bsv3 files **(not recommended)** you can disable it with the argument --disable_bsv3.
Disabling the bsv3 processing will prevent the 'convert' tool from generating the frames of the animations and instead it will generate only a single image.

```
python3 -m tstorgb.convert --search_zip --disable_bsv3 rgb_dir destination
```

Again, if you use this argument you might not get the results you were expecting.
