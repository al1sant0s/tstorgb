# Convert any RGB files from TSTO

This package allows you to convert the raw RGB assets from 'The Simpsons: Tapped Out' game into proper images.
It uses [**libvips**](https://www.libvips.org/) to perform the conversions.

## Currently supported files

  * ✅ **rgb**
  * ✅ **bcell**
  * ✅ **bsv3**

## Installation

First, make sure you install [**libvips**](https://www.libvips.org/install.html) in your system. Then, run the
following command in the command line:

```
python3 -m pip install tstorgb@git+https://github.com/al1sant0s/tstorgb
```

## Usage

```
tstorgb --help
```

The convert tool will receive a list of directories to search for the rgb files, convert them into images of specified format and then
it will save the results in the last directory you provide. For example, supposing the rgb files are inside a directory called 'rgb_dir' and you want the images to be exported to the 'destination' directory, you would issue the following command:

```
tstorgb path/to/rgb_dir path/to/destination
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

In theory, the entity is just the name that precedes the first underscore character in a filename and the variation is the rest after the underscore excluding the .rgb extension.
For example, given the name 'something_anything_else.rgb', _something_ is the entity and _anything_else_ is the variation.
When a file corresponding to an entity doesn't specify a variation (a filename without an underscore in it like 'orangehouse.rgb') the _default variation subdirectory will be created.

## Arguments

If you have your rgb files inside a zip file named '1', you can pass the --search_zip argument to deal with the extraction.

```
tstorgb --search_zip path/to/rgb_dir path/to/destination
```

If you prefer to use a file extension other than png, you can use the --output_extension argument.

```
tstorgb --search_zip --output_extension webp path/to/rgb_dir path/to/destination
```

## Multiple directories

An example specifying multiple directories as input and saving the results in the sprites directory.

```
tstorgb --search_zip path/to/Downloads/rgb_files path/to/Images/rgb_folder path/to/Images/another_rgb_folder path/to/sprites  
```

If you want to search for the rgb files recursively in every subdirectory bellow a specific directory, give the root directory as input. The following example will convert all the rgb files within the 'Images' directory,
including the 'rgb_folder' and 'another_rgb_folder' subdirectories.

```
tstorgb --search_zip path/to/Downloads/rgb_files path/to/Images path/to/sprites  
```
