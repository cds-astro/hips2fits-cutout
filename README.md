# hips2fits-cutout

HiPS2FITS cutout script - Cutouts generation from local HiPS datasets

## Usage

### Generating a single cutout

General syntax is:

`./hips2fits_cutout.py <ra> <dec> <fov> <width> <height> <hips_path> <output_path> [img_format] [stretch]`

Compulsory parameters are:
- ra: right ascension in decimal degrees of the cutout center
- dec: declination in decimal degrees of the cutout center
- fov: size in decimal degrees of the largest dimension of the cutout
- width: in pixels
- height: in pixels
- hips_path: local path to HiPS root
- output_path: path of the cutout output

Two optional parameters are:
- img_format: *fits*, *jpg* or *png*. Default is *fits*
- stretch: (applicable for PNG or JPG cutouts). Possible values are *pow2*, *linear*, *sqrt*, *log*, *asinh*. Default is *linear*.

Example:

`./hips2fits_cutout.py 83.6287 22.0147 0.2 1000 800 /path/to/hips/root my_cutout.fits`


## License

HiPS2FITS cutout script is distributed under the LGPL-3.0 license.

## Acknowledgment

If he HiPS2FITS cutout script was useful for your research, please use this acknowledgment:

`This research has made use of the HiPS2FITS cutout script, CDS, Strasbourg Astronomical Observatory, France (DOI : 10.26093/2msf-n437).`

