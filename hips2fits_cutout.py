#!/usr/bin/env python

 ##############################################################################
 #
 # Copyright 2024 - Thomas Boch (CDS)
 #
 # This file is part of HiPS2FITS cutout script.
 #
 # HiPS2FITS cutout script is free software: you can redistribute it and/or modify
 # it under the terms of the GNU Lesser General Public License as published by
 # the Free Software Foundation, either version 3 of the License, or
 # (at your option) any later version.
 #
 # HiPS2FITS cutout script is distributed in the hope that it will be useful,
 # but WITHOUT ANY WARRANTY; without even the implied warranty of
 # MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 # GNU Lesser General Public License for more details.
 #
 # You should have received a copy of the GNU Lesser General Public License
 # along with HiPS2FITS cutout script.  If not, see <http://www.gnu.org/licenses/>.
 #
 ##############################################################################


import sys
import os

from PIL import Image


from astropy.table import Table

from io import BytesIO

import time

import re

import math

from functools import lru_cache

import numpy as np
from multiprocessing import Pool

import matplotlib.image as mimg

from astropy.coordinates import Longitude, Latitude

from astropy.visualization import simple_norm

import numba
numba.config.NUMBA_NUM_THREADS = max(1, os.cpu_count() // 3 - 1)
from numba.typed import Dict

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, Angle

import numpy as np

from astropy.io import fits

from astropy.wcs.utils import proj_plane_pixel_scales

import cdshealpix

import astropy.units as u



DEFAULT_FORMAT = 'fits'
DEFAULT_STRETCH = 'linear'
DEFAULT_CMAP = 'Greys_r'


@lru_cache(maxsize=None)
def _compute_xy2hpx(shift_order: int) -> np.ndarray:
    """
    TODO: write a good description what this is.
    Note that there is already a mention of this in the high-level docs:
    https://github.com/hipspy/hips/blame/71e593ab7e60767be70d9b2b13398016c35db09a/docs/drawing_algo.rst#L103
    Parameters
    ----------
    shift_order : int
        The HiPS tile "shift order", which is related to the tile
        pixel width as follows: ``tile_width = 2 ** shift_order``.
        Supported range of values: 1 to 16
    Returns
    -------
    shift_ipix_array : `~numpy.ndarray`
        2-dimensional array of HEALPix nested order ``ipix`` values
        for the tile pixels. These numbers are relative to the
        HiPS tile HEALPix index, which needs to be added.
    Examples
    --------
    TODO: give examples here, or elsewhere from where this helper is called?
    """
    # Sanity check, prevent users from shooting themselves in the foot here
    # and waste a lot of CPU and memory.
    if not isinstance(shift_order, int):
        raise TypeError('The `shift_order` option must by of type `int`.')
    # Usually tiles have ``shift_order == 9``, i.e. ``tile_width == 512`
    # There's no examples for very high shift order,
    # so the ``shift_oder == 16`` limit (``tile_width == 65536``) here should be OK.
    if shift_order < 1 or shift_order > 16:
        raise ValueError('The `shift_order` must be in the range 1 to 16.')

    if shift_order == 1:
        return np.array([[0, 1], [2, 3]])
    else:
        # Create 4 tiled copies of the parent
        ipix_parent = _compute_xy2hpx(shift_order - 1)
        data1 = np.tile(ipix_parent, reps=(2, 2))

        # Add the right offset values to each of the 4 parts
        repeats = 2 ** (shift_order - 1)
        data2 = (repeats ** 2) * np.array([[0, 1], [2, 3]])
        data2 = np.repeat(data2, repeats, axis=0)
        data2 = np.repeat(data2, repeats, axis=1)

    return data1 + data2

@lru_cache(maxsize=None)
def _compute_hpx2xy(shift_order):
    xy2hpx = _compute_xy2hpx(shift_order)
    flattened_xy2hpx = xy2hpx.flatten()
    tile_width = xy2hpx.shape[0]
    hpx2xy = np.empty([tile_width**2, 2], dtype=np.int32)
    for i in range(0, tile_width**2):
        x = i // tile_width
        y = i - x*tile_width

        hpx2xy[flattened_xy2hpx[i]] = [x, y]

    return hpx2xy

def compute_interpolation_coeff(lon, lat, order):
    return cdshealpix.nested.bilinear_interpolation(lon, lat, order, num_threads=32)


def compute_pix2world(wcs, x, y, hips_frame='equatorial'):
    skycoords = wcs.pixel_to_world(x, y)
    # does not happen to be faster :-(
    #skycoords = pixel_to_skycoord(x, y, wcs)
    if hips_frame=='equatorial':
        return skycoords.icrs.ra, skycoords.icrs.dec
    elif hips_frame=='galactic':
        return skycoords.galactic.l, skycoords.galactic.b
    else:
        # planetary case!
        return skycoords.icrs.ra, skycoords.icrs.dec

def _create_wcs_object(skycoord, width, height, fov, coordsys='icrs', projection='SIN', rotation_angle=0, inverse_longitude=False):
    """
    Create as Astropy WCS object from a few basic parameters

    Largely inspired by WCSGeometry.create method of hips package, added ability to specify rotation angle
    """

    wcs_tmp = WCS(header={'CRVAL1': 0.0, 'CRVAL2': 0.0, 'CRPIX1': 0.0, 'CRPIX2': 0.0,
                          'CTYPE1': 'RA---' + projection, 'CTYPE2': 'DEC--' + projection,
                          'CD1_1': 1.0, 'CD1_2': 0.0, 'CD2_1': 0.0, 'CD2_2': 1.0})
    fov_standard = 2 * wcs_tmp.wcs_world2pix(fov / 2.0, 0.0, 1)[0]

    fov = Angle(fov_standard, unit='degree')

    crpix = float(width / 2), float(height / 2)
    cdelt = float(fov.degree) / float(max(width, height))

    header = {}
    header['NAXIS'] = 2

    if coordsys == 'icrs':
        header['CTYPE1'] = f'RA---{projection}'
        header['CTYPE2'] = f'DEC--{projection}'
        header['CRVAL1'] = skycoord.icrs.ra.deg
        header['CRVAL2'] = skycoord.icrs.dec.deg
    elif coordsys == 'galactic':
        header['CTYPE1'] = f'GLON-{projection}'
        header['CTYPE2'] = f'GLAT-{projection}'
        header['CRVAL1'] = skycoord.galactic.l.deg
        header['CRVAL2'] = skycoord.galactic.b.deg
    else:
        raise ValueError('Unrecognized coordinate system.')

    header['CRPIX1'] = crpix[0]
    header['CRPIX2'] = crpix[1]


    longitude_sign = 1
    if inverse_longitude:
        longitude_sign = -1
    if rotation_angle != 0:
        header['CD1_1'] =  -cdelt * np.cos(rotation_angle * np.pi / 180.) * longitude_sign
        header['CD1_2'] =  -cdelt * np.sin(rotation_angle * np.pi / 180.)
        header['CD2_1'] =  -cdelt * np.sin(rotation_angle * np.pi / 180.)
        header['CD2_2'] =  cdelt * np.cos(rotation_angle * np.pi / 180.)
    else:
        header['CDELT1'] = -cdelt * longitude_sign
        header['CDELT2'] = cdelt

    header['NAXIS1'] = width
    header['NAXIS2'] = height

    return WCS(header=header)

def _get_image_scale(wcs):
    """
    return resolution in degree/pixel, from a WCS object
    """
    return min(proj_plane_pixel_scales(wcs))

def _get_healpix_order_for_resolution(resolution):
    """
    retrieve healpix order for resolution (expressed in degree)
    The optimal order is the one so that the resolution is equal or slightly better than the input resolution
    """
    for order in range(0, 30):
        hpx_res = math.degrees(np.sqrt(4 * np.pi / (12*4**order)))
        if hpx_res <= resolution:
            return order

    return 29

def _parse_properties_as_dict(properties_path):
    props = {}
    with open(properties_path) as h:
        while True:
            l = h.readline()
            if not l:
                break

            l = l.rstrip()
            m = re.search('(.*?)=(.*)', l)
            if m:
                props[m.group(1).strip()] = m.group(2).strip()

    return props

def _get_tile_path(root_url, norder, npix, img_format):
        """
        return URL path for tile norder, npix for HiPS at root_url
        """
        dir_nb = int((npix // 10000) * 10000)
        format = img_format.lower().replace('jpeg', 'jpg')
        return f'{root_url}/Norder{norder}/Dir{dir_nb}/Npix{npix}.{format}'

def _get_allsky_tile_path(root_url, img_format):
        """
        return URL path for tile norder, npix for HiPS at root_url
        """
        format = img_format.lower().replace('jpeg', 'jpg')

        return f'{root_url}/Norder3/Allsky.{format}'

def make_cutout(width, height, wcs, hips_root, coordsys='icrs', tile_format='fits'):
    PARALLELISM_LEVEL = 8 # number of concurrent processes

    hips_properties = _parse_properties_as_dict(os.path.join(hips_root, 'properties'))
    hips_frame = hips_properties.get('hips_frame', 'icrs')

    ### 1st step: compute sky location for each pixel
    #             of the output image
    # TODO: is there a 0.5 shift to be taken into account???
    t1_start = time.time()
    xv, yv = np.meshgrid(np.arange(0, width), np.arange(0, height))

    if width*height >= 4e6:
        use_processes = True
    else:
        use_processes = False

    if use_processes:
        nb_sections = max(1, width*height // 100000)

        xv_splitted = np.array_split(xv, nb_sections)
        yv_splitted = np.array_split(yv, nb_sections)

        pool = Pool(processes=PARALLELISM_LEVEL)
        input_list = []
        for i in range(len(xv_splitted)):
            input_list.append([wcs, xv_splitted[i], yv_splitted[i], hips_frame])

        lon, lat = np.concatenate(pool.starmap(compute_pix2world, input_list), axis=1)
        lon = Longitude(lon, unit='deg')
        lat = Latitude(lat, unit='deg')

        pool.close()

    else:
        lon, lat = compute_pix2world(wcs, xv, yv, hips_frame)

    t1_end = time.time()
    #print(f'T1: {t1_end-t1_start}')



    ### 2nd step: compute HEALPix indexes for each sky location (bilinear interpolation)
    t2_start = time.time()
    output_resolution = _get_image_scale(wcs)
    hips_order = int(hips_properties['hips_order'])
    tile_size = 'hips_tile_width' in hips_properties and int(hips_properties['hips_tile_width']) or 512
    if tile_size<=0:
        tile_size = 512
    shift_order = int(np.log2(tile_size))

    tile_order = _get_healpix_order_for_resolution(output_resolution) - shift_order
    if tile_order<0:
        tile_order = 0


    tile_order = min(hips_order, tile_order) # we can't go deeper than hips_order
    hips_ID = hips_properties.get('ID', None)
    if hips_ID is None:
        hips_ID = hips_properties.get('creator_did', None)

    # tile_order can't be lower than hips_order_min
    if 'hips_order_min' in hips_properties:
        tile_order = max(tile_order, int(hips_properties['hips_order_min']))


    pixel_order = tile_order + shift_order


    coeffs = compute_interpolation_coeff(lon, lat, pixel_order)

    coeffs_0 = coeffs[0].filled(fill_value = -1)
    coeffs_1 = coeffs[1].filled(fill_value = -1)

    t2_end = time.time()
    #print(f'T2: {t2_end-t2_start}')


    ### 3rd step: apply bilinear interpolation
    t3_start = time.time()
    hpx2xy = _compute_hpx2xy(shift_order)

    borders_lon = Longitude([lon[0][0], lon[height-1][0], lon[height-1][width-1], lon[0][width-1]], unit='deg')
    borders_lat = Latitude([lat[0][0], lat[height-1][0], lat[height-1][width-1], lat[0][width-1]], unit='deg')

    estimated_fov = output_resolution * max(width, height)

    if np.any(np.isnan(lon)) or np.any(np.isnan(lat)) or estimated_fov>180:
        ipixes, orders, fully_covered = cdshealpix.cone_search(0.*u.deg, 0.*u.deg, 180.*u.deg, tile_order, flat=True)
    else:
        ipixes, orders, fully_covered = cdshealpix.polygon_search(borders_lon, borders_lat, tile_order, flat=True)

    tiles = {}
    numpy_data_type = None
    numba_data_type = None

    # find types from Allsky tiles
    allsky_path = _get_allsky_tile_path(hips_root, tile_format)



    if tile_format=='fits':
        allsky_data = None
        # is it a RICE compressed tile?
        if not allsky_path.startswith('http'):
            allskypath_fz = allsky_path + '.fz'
            if os.path.exists(allskypath_fz):
                allsky_data = fits.open(allskypath_fz)[1].data

        if allsky_data is None:
            allsky_data = fits.open(allsky_path)[0].data
    else:
        if allsky_path.startswith('http'):
            r = requests.get(allsky_path)
            with Image.open(BytesIO(r.content)) as image:
                allsky_data = np.array(image)
        else:
            with Image.open(allsky_path) as image:
                allsky_data = np.array(image)

    numpy_data_type = allsky_data.dtype
    numba_data_type = numba.typeof(allsky_data[0][0])

    for ipix in ipixes:
        ipix = ipix.item()
        hdu = None
        tile_path = _get_tile_path(hips_root, tile_order, ipix, tile_format)
        data = _get_image_data(tile_path, tile_size)

        if numpy_data_type.str.endswith('f4'): # struggling with numpy and numba types ...
            tiles[ipix] = data.astype(np.float32)
            numpy_data_type = tiles[ipix].dtype
        elif numpy_data_type.str.endswith('f8'):
            tiles[ipix] = data.astype(np.float64)
            numpy_data_type = tiles[ipix].dtype
        elif numpy_data_type.str.endswith('i2'):
            tiles[ipix] = data.astype(np.int16)
            numpy_data_type = tiles[ipix].dtype
        elif numpy_data_type.str.endswith('i4'):
            tiles[ipix] = data.astype(np.int32)
            numpy_data_type = tiles[ipix].dtype
        elif numpy_data_type.str == '<u2':
            tiles[ipix] = data.astype(np.int16)
            numpy_data_type = tiles[ipix].dtype
            numba_data_type = numba.typeof(tiles[ipix][0][0])
        else:
            tiles[ipix] = data


    if tile_format=='fits':
        value_type = numba_data_type[:,:]
    else:
        value_type = numba.types.uint8[:,:,:]

    dict_tiles = Dict.empty(
        key_type=numba.types.int64,
        value_type=value_type
    )
    for idx in tiles.keys():
        dict_tiles[idx] = tiles[idx]

    if tile_format=='fits':
        cutout = dispatch_weights_to_pixels_fits(xv, yv, dict_tiles, coeffs_0, coeffs_1, hpx2xy, numpy_data_type)
    else:
        cutout = dispatch_weights_to_pixels_jpg(xv, yv, dict_tiles, coeffs_0, coeffs_1, hpx2xy, numpy_data_type, tile_format)

    t3_end = time.time()
    #print(f'T3: {t3_end-t3_start}')
    # TODO: mask pixels outside projection

    return cutout

@numba.jit(nopython=True, nogil=True, parallel=False, fastmath=True, cache=True)
def dispatch_weights_to_pixels_fits(xv, yv, dict_tiles, interp_ipix, interp_weight, hpx2xy, numpy_data_type):
    w = interp_ipix.shape[0]
    h = interp_ipix.shape[1]
    tile_width = int(np.sqrt(len(hpx2xy)))
    shift_order = int(np.log2(tile_width))

    out = np.zeros((w, h), dtype=numpy_data_type)

    for x in numba.prange(w):
        for y in numba.prange(h):
            weights = interp_weight[x][y]
            ipixes = interp_ipix[x][y]

            val = 0

            for i in numba.prange(4):
                if weights[i]<=0:
                    continue
                ipix = int(ipixes[i])
                ipix_tile = ipix // 4**shift_order

                xx, yy = hpx2xy[ipix - ipix_tile*4**shift_order]

                if dict_tiles.get(ipix_tile) is None:
                    continue
                pix_val = dict_tiles[ipix_tile][tile_width-yy-1][xx]

                val = val + weights[i] * pix_val

            out[x, y] = val

    return out

# I had to copy most of the code from dispatch_weights_to_pixels_fits because numba did not want to
# compile a more generic version of the code
@numba.jit(nopython=True, nogil=True, parallel=False, fastmath=True, cache=True)
def dispatch_weights_to_pixels_jpg(xv, yv, dict_tiles, interp_ipix, interp_weight, hpx2xy, numpy_data_type, tile_format):
    w = interp_ipix.shape[0]
    h = interp_ipix.shape[1]
    tile_width = int(np.sqrt(len(hpx2xy)))
    shift_order = int(np.log2(tile_width))

    n_dimensions = 3
    if tile_format=='png':
        n_dimensions = 4

    out = np.zeros((w, h, n_dimensions), dtype=numpy_data_type)

    for x in numba.prange(w):
        for y in numba.prange(h):
            weights = interp_weight[x][y]
            ipixes = interp_ipix[x][y]

            val = np.zeros(n_dimensions)

            for i in numba.prange(4):
                ipix = int(ipixes[i])
                ipix_tile = ipix // 4**shift_order

                xx, yy = hpx2xy[ipix - ipix_tile*4**shift_order]

                #pix_val = dict_tiles[ipix_tile][tile_width-yy-1][xx]
                if dict_tiles.get(ipix_tile) is None:
                    continue
                pix_val = dict_tiles[ipix_tile][yy][xx]

                val += weights[i] * pix_val

            out[x, y] = val


    return out

def _apply_stretch(input_image, stretch='linear', min_cut=None, max_cut=None, asinh_a=None):
    if not asinh_a:
        asinh_a = 0.1

    image_normalizer = simple_norm(input_image, stretch=stretch, min_cut=min_cut, max_cut=max_cut, asinh_a=asinh_a, clip=True)
    image_scaled = image_normalizer(input_image)
    image_scaled = np.flipud(image_scaled)

    return image_scaled

def _make_jpg_or_png(output_file, cutout, min_cut, max_cut, stretch, cmap, cutout_img_format, hips_properties):
    image_scaled = _make_scaled_image(cutout, stretch, min_cut, max_cut, hips_properties)

    mimg.imsave(output_file, image_scaled, format=cutout_img_format, cmap=cmap, vmin=0, vmax=1, dpi=42) # specifying the dpi is a workaround around bug https://github.com/matplotlib/matplotlib/issues/13253

    # this helps garbage collection on some configs
    del image_scaled

def _make_scaled_image(cutout, stretch, min_cut, max_cut, hips_properties):
    if min_cut is not None:
        min_cut_str = str(min_cut)
        if str(min_cut).endswith('%'):
            min_cut = np.nanpercentile(cutout, float(min_cut_str[:-1]))
        else:
            min_cut = float(min_cut)
    else:
        if hips_properties.get('dataproduct_subtype', '')=='color':
            min_cut = 0
        else:
            min_cut = np.nanpercentile(cutout, 0.5)

    if max_cut is not None:
        max_cut_str = str(max_cut)
        if max_cut_str.endswith('%'):
            max_cut = np.nanpercentile(cutout, float(max_cut_str[:-1]))
        else:
            max_cut = float(max_cut)
    else:
        if hips_properties.get('dataproduct_subtype', '')=='color':
            max_cut = 255
        else:
            max_cut = np.nanpercentile(cutout, 99.5)

    return _apply_stretch(cutout, stretch=stretch, min_cut=min_cut, max_cut=max_cut)

def _get_image_data(fits_path_or_url, tile_size):
    # jpg or png tiles
    if not fits_path_or_url.endswith('fits'):
        n_dim = 3
        if fits_path_or_url.endswith('png'):
            n_dim = 4

        data = np.empty((tile_size, tile_size, n_dim)).astype(np.uint8)

        try:
            if fits_path_or_url.startswith('http'):
                r = requests.get(fits_path_or_url)
                with Image.open(BytesIO(r.content)) as image:
                    data = np.array(image)
            elif os.path.exists(fits_path_or_url):
                with Image.open(fits_path_or_url) as image:
                    data = np.array(image)

        except Exception as e:
            logging.error(str(e))

        return data

    data = np.full((tile_size, tile_size), np.nan)
    if fits_path_or_url.startswith('http') or os.path.exists(fits_path_or_url):
        try:
            hdu = fits.open(fits_path_or_url, do_not_scale_image_data=True)
        except:
            return data

        data = hdu[0].data
        hdr  = hdu[0].header

        blank_value = None
        if 'BLANK' in hdr:
            blank_value = hdr['BLANK']
            blank_mask = np.where(data==blank_value)

        if 'BSCALE' in hdr or 'BZERO' in hdr:
            bzero = 0.0
            bscale = 1.0
            if 'BZERO' in hdr:
                bzero = hdr['BZERO']
            if 'BSCALE' in hdr:
                bscale = hdr['BSCALE']

            data = bscale * data + bzero
            if blank_value:
                data[blank_mask] = np.nan

    else:
        # RICE compressed .fits.fz tile?
        if not fits_path_or_url.startswith('http'):
            fz_path = fits_path_or_url + '.fz'
            if os.path.exists(fz_path):
                data = fits.open(fz_path)[1].data

    return data

def generate_from_wcs(wcs, hips_path, output_path, format='fits', min_cut=None, max_cut=None,
                                                                  stretch='linear', cmap=DEFAULT_CMAP):
    start = time.time()
    
    hips_properties = _parse_properties_as_dict(os.path.join(hips_path, 'properties'))
    is_color_hips = hips_properties.get('dataproduct_subtype', '')=='color'

    tile_format = 'fits'
    if is_color_hips:
        if hips_properties['hips_tile_format'].find('png')>=0:
            tile_format = 'png'
        else:
            tile_format = 'jpg'

    width, height = wcs.pixel_shape
    cutout = make_cutout(width, height, wcs, hips_path, tile_format=tile_format)

    fits_header = wcs.to_header()
    fits_header.add_history('Generated by hips2fits-cutout script')
    fits_header.add_history('From HiPS {} ({})'.format(hips_properties.get('ID', ''), hips_properties.get('obs_title', '')))
    if 'hips_creator' in hips_properties:
        fits_header.add_history('HiPS created by ' + hips_properties.get('hips_creator', '') + ' - ' + hips_properties.get('hips_copyright', ''))

    copyright_str = hips_properties.get('obs_copyright', '')
    if copyright_str != '':
        if 'obs_copyright_url' in hips_properties:
            copyright_str += ' - ' + hips_properties['obs_copyright_url']
            fits_header.set('CPYRIGHT', copyright_str)


    if format=='fits':
        if tile_format != 'fits': # in that case, we need to rearrange the array, first dimension becomes last one
            cutout = np.moveaxis(cutout, 2, 0)

        fits.writeto(output_path, data=cutout, header=fits_header, overwrite=True)
    else:
        _make_jpg_or_png(output_path, cutout, min_cut, max_cut, stretch, cmap, format, hips_properties)

    end = time.time()

    print(f'Cutout {output_path} generated in {end-start:.2f}s')   

    # TODO: return stat (generation time, success/error, errorCause) 

def generate(ra, dec, fov, width, height, hips_path, output_path, format='fits', min_cut=None, max_cut=None,
                                                                  stretch='linear', cmap=DEFAULT_CMAP):

    sc = SkyCoord(ra, dec, frame='icrs', unit='deg')
    wcs = _create_wcs_object(sc, width, height, fov, coordsys='icrs', projection='SIN', rotation_angle=0)

    generate_from_wcs(wcs, hips_path, output_path, format=format, min_cut=min_cut, max_cut=max_cut, stretch=stretch, cmap=cmap)



def generate_for_list(params_table, min_cut=None, max_cut=None, stretch='linear', cmap='Greys'):
    start = time.time()

    # TODO: check column names. If no column names, use default order
    # TODO: additional params : cmap, min/max_cut, stretch, format

    for row in params_table:
        format = DEFAULT_FORMAT
        min_cut = max_cut = None
        stretch = DEFAULT_STRETCH
        cmap = DEFAULT_CMAP
        if 'format' in params_table.colnames:  format = row['format']
        if 'min_cut' in params_table.colnames: min_cut = row['min_cut']
        if 'max_cut' in params_table.colnames: max_cut = row['max_cut']
        if 'stretch' in params_table.colnames: stretch = row['stretch']
        if 'cmap' in params_table.colnames:    cmap = row['cmap']

        generate(row['ra'], row['dec'], row['fov'], row['width'], row['height'], row['hips'], row['output'],
                     format=format, min_cut=min_cut, max_cut=max_cut, stretch=stretch, cmap=cmap)

    # return stat (generation time, success/error, errorCause)

    # TODO: check params column names

    end = time.time()

    print(f'\n\n{len(params_table)} cutouts generated in {end-start:.2f} seconds')

    # return array of stat (generation time, success/error, errorCause)

def create_html_page(params_table, html_path, link_template):
    with open(html_path, 'w') as h:
        h.write('<html>\n')
        h.write('  <head>\n')
        h.write('    <style>\n')
        h.write('      .imgHolder { display: inline-block; position: relative; margin: 0;}\n')
        h.write('      .caption { position: absolute; bottom: 3px; left: 2px; font-size: 0.9em; font-family: "Helvetica Neue",Helvetica,Arial,sans-serif; background: rgba(255, 255, 255, 0.6); padding: 2px; z-index: 3; max-width: 99%; overflow: hidden; text-overflow: ellipsis; }\n')
        h.write('    </style>\n')
        h.write('  </head>\n')
        h.write('  <body>\n')
        for row in params_table:
            ra  = row['ra']
            dec = row['dec']
            fov = row['fov']
            thumb_path = row['output']
            label = None
            title = thumb_path
            if 'label' in params_table.colnames:
                label = row['label']

            h.write('    <div class="imgHolder">\n')
            if link_template:
                url = link_template.format(ra=row['ra'], dec=row['dec'], fov=row['fov'])
                h.write(f'    <a href="{url}" target="_blank">')
            h.write(f'      <img loading="lazy" src="{thumb_path}" title="{title}">\n')
            if link_template:
                h.write('    </a>')
            if label:
                h.write(f'      <div class="caption">{label}</div>\n')
            h.write('    </div>\n')
        h.write('  </body>\n')
        h.write('</html>')


if __name__ == '__main__':
    start = time.time()

    if '-l' in sys.argv or '--list-params' in sys.argv:
        if '-l' in sys.argv:
            list_params_path_idx = sys.argv.index('-l')
        else:
            list_params_path_idx = sys.argv.index('--list-params')

        list_params_path = sys.argv[list_params_path_idx + 1]
        params = Table.read(list_params_path, format='csv')

        if '-html' in sys.argv:
            html_path = sys.argv[sys.argv.index('-html') + 1]

            link_template = None
            if '--link-template' in sys.argv:
                link_template = sys.argv[sys.argv.index('--link-template') + 1]

            create_html_page(params, html_path, link_template)

        else:
            generate_for_list(params)

        sys.exit()


    start = time.time()

    ra = float(sys.argv[1])
    dec = float(sys.argv[2])
    fov = float(sys.argv[3])
    width  = int(sys.argv[4])
    height = int(sys.argv[5])
    hips_path = sys.argv[6]
    output_path = sys.argv[7]

    img_format = 'fits'
    if len(sys.argv)>8:
        img_format = sys.argv[8]

    stretch = 'linear'
    if len(sys.argv)>9:
        stretch = sys.argv[9]

    generate(ra, dec, fov, width, height, hips_path, output_path, format=img_format, stretch=stretch)

    end = time.time()
    print(f'Generated in {end-start} s')
