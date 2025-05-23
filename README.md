# Convert any RGB files from TSTO

This package allows you to convert the raw RGB assets from 'The Simpsons: Tapped Out' game into proper images.
It uses [**libvips**](https://www.libvips.org/) to perform the conversions.

## Currently supported files

  * ✅ **rgb**
  * ✅ **bcell**
  * ✅ **bsv3**

## Installation

First, make sure you install [**libvips**](https://www.libvips.org/install.html) in your system.

> Attention, windows users!
> 
> When downloading libvips make sure to update the [**PATH environment variable**](https://learn.microsoft.com/en-us/previous-versions/office/developer/sharepoint-2010/ee537574(v=office.14)) to point to the **vips-x.y.z/bin** directory (remember you have to specify the full path). That is after you unzip the downloaded file as instructed in the installation page.
If you don't set the PATH environment variable correctly, the tool will not find the required dlls and thus it won't work.

You'll also need to install [**python**](https://www.python.org/downloads/)
and [**git**](https://git-scm.com/downloads) if you don't already have them installed in your system.

With everything ready, run either of the following commands in the command-line interface, according to your OS:

* Windows installation command.
```
python -m pip install tstorgb@git+https://github.com/al1sant0s/tstorgb
```
* Linux installation command.
```
python3 -m pip install tstorgb@git+https://github.com/al1sant0s/tstorgb
```

If you use windows I recommend you to get the modern [windows terminal from microsoft store](https://apps.microsoft.com/detail/9n0dx20hk701?hl).

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

## Bsv3 and Bcell files

To process bsv3 and bcell files just keep them in the same directory as their correspondent rgb files. The correspondent files have the same names before the extension, like building.rgb and building.bsv3 or character_does_action_image_x.rgb and character_does_action.bcell. As long as they are in the same directories, they'll be automatically processed. You can prevent their processing by passing the flags --disable_bsv3 and –disable_bcell. 

## Animated images

The --sequential and --sequential_delay arguments make it possible to produce animated images using specific image formats (e.g. webp). To produce animated images
with a refresh rate of 30 fps, for example:

```
tstorgb --search_zip --sequential --sequential_delay 33 --output_extension webp path/to/Images path/to/sprites
```

## Grouping images

Set the --group argument to organize the files in subdirectories within the destination directory.
Each subdirectory corresponds to a certain entity. For example, if there were two files in 'rgb_dir', one named
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

In theory, the entity is just the name that precedes the first underscore character in a filename and the variation
is the rest after the underscore excluding the file extension.
For example, given the name 'something_anything_else.rgb', _something_ is the entity and _anything_else_ is the variation.
When a file corresponding to an entity doesn't specify a variation (a filename without an underscore in it like 'orangehouse.rgb') the _default variation subdirectory will be created.

```
tstorgb --group path/to/rgb_dir path/to/destination
```

## Short options


Here is a list of some options with their correspondent short options.

* --search_zip [-z]
```shell
tstorgb -z path/to/Images path/to/sprites
```

* --image_quality [-q]
```shell
tstorgb -q 95 path/to/Images path/to/sprites
```

* --output_extension [-e]
 ```shell
tstorgb -e webp path/to/Images path/to/sprites
```

* --group [-g]
```shell
tstorgb -g path/to/Images path/to/sprites
```
