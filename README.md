# hips2fits-cutout

HiPS2FITS cutout script - Cutouts generation from local HiPS datasets

## Dependencies

Libraries to be installed are described in [*requirements.txt*](https://github.com/cds-astro/hips2fits-cutout/blob/main/requirements.txt).

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
- stretch: (applicable for PNG or JPG cutouts). Possible values are *pow*, *linear*, *sqrt*, *log*, *asinh*. Default is *linear*.

Example:

`./hips2fits_cutout.py 83.6287 22.0147 0.2 1000 800 /path/to/hips/root my_cutout.fits`

### Generating several cutouts from a list of positions

The script can be used from the command line:
```bash
./hips2fits_cutout.py -l <csv-params-table>
```

You will find an example of such a parameters file in [*params-SDSS.csv*](https://github.com/cds-astro/hips2fits-cutout/blob/main/params-SDSS.csv)

It can also be used directly from your Python script:
```python
from astropy.table import Table
from hips2fits_cutout import generate_for_list

params = Table.read('params-SDSS.csv', format='csv')
generate_for_list(params)
```

The parameter file must include at least the following columns:

`ra, dec, fov, width, height, hips, output, format`

You can add optional columns: *min_cut*, *max_cut*, *stretch*

*stretch* can have the following values: 'pow', 'linear', 'sqrt', 'asinh', 'log'
*min_cut* or *max_cut* can be given as an absolute value in the form of a float, or as a percentile (e.g., '99.97%').

## License

HiPS2FITS cutout script is distributed under the LGPL-3.0 license.

## Acknowledgment

If he HiPS2FITS cutout script was useful for your research, please use this acknowledgment:

`This research has made use of the HiPS2FITS cutout script, CDS, Strasbourg Astronomical Observatory, France (DOI : 10.26093/2msf-n437).`

