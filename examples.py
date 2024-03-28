#!/usr/bin/env python

from hips2fits_cutout import generate, generate_from_wcs

from astropy.io import fits
from astropy.wcs import WCS

hips_path = 'hips-F658N'

width_pixels = 1500
height_pixels = 1500

# 1. generate a cutout from simple parameters: image center (ra, dec), size on the sky (fov)
generate(84.66945, -69.09788, 0.04, width_pixels, height_pixels, hips_path, 'cutout1.fits', format='fits')

# 2. generate a cutout from an astropy WCS object
hdu = fits.open('2MASS.fits')
wcs = WCS(hdu[0].header)
print(wcs)
generate_from_wcs(wcs, hips_path, 'cutout2.fits', format='fits')
