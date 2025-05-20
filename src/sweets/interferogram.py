#!/usr/bin/env python
from __future__ import annotations

import logging
from functools import partial
from os import fsdecode
from pathlib import Path
from typing import Optional

import jax.numpy as jnp
import numpy as np
from jax import jit, lax
from dolphin import io, Strides
from opera_utils import get_dates

logger = logging.getLogger(__name__)


def create_interferogram(
    ref_slc_file: Path | str,
    sec_slc_file: Path | str,
    *,
    outfile: Path | str | None = None,
    dset: str | None = None,
    looks: tuple[int, int],
    overwrite: bool = False,
    block_shape: tuple[int, int] = (1024, 1024),
) -> Path:
    """Create a multi-looked, normalized interferogram from GDAL-readable SLCs.

    Parameters
    ----------
    ref_slc_file : Path | str
        Path to reference SLC.
    sec_slc_file : Path | str
        Path to secondary SLC.
    outfile : Path | str
        Path to output file.
    looks : tuple[int, int]
        row looks, column looks.
    overwrite : bool, optional
        Overwrite existing interferogram in `outfile`, by default False.

    Returns
    -------
    Path
        Path to Geotiff file containing the multi-looked, normalized interferogram.
    """
    out_path = (
        _form_ifg_name(ref_slc_file, sec_slc_file, Path())
        if outfile is None
        else Path(outfile)
    )

    if out_path.exists():
        if not overwrite:
            logger.debug(f"Skipping {out_path} because it already exists.")
            return out_path
        else:
            logger.info(f"Overwriting {out_path} because overwrite=True.")
            out_path.unlink()

    def _get_gdal_str(filename) -> str:
        if (
            "hdf5:" in fsdecode(filename).lower()
            or "netcdf:" in fsdecode(filename).lower()
        ):
            return fsdecode(filename)
        else:
            return io.format_nc_filename(filename, dset)

    gdal_ref_slc_file = _get_gdal_str(ref_slc_file)
    gdal_sec_slc_file = _get_gdal_str(sec_slc_file)
    crs = io.get_raster_crs(gdal_ref_slc_file)
    assert crs == io.get_raster_crs(gdal_sec_slc_file)

    logger.info(
        f"Creating {looks[0]}x{looks[1]} multi-looked interferogram: {out_path}"
    )
    reader_ref = io.RasterReader.from_file(gdal_ref_slc_file, nodata=np.nan)
    reader_sec = io.RasterReader.from_file(gdal_sec_slc_file, nodata=np.nan)
    # reader_ref = io.HDF5Reader(filename=ref_slc_file, dset_name=dset, nodata=np.nan)
    # reader_sec = io.HDF5Reader(filename=sec_slc_file, dset_name=dset, nodata=np.nan)

    io.write_arr(
        arr=None,
        output_name=out_path,
        like_filename=gdal_ref_slc_file,
        strides={"y": looks[0], "x": looks[1]},
        nodata=0,
    )

    block_manager = io.StridedBlockManager(
        arr_shape=reader_ref.shape, block_shape=block_shape, strides=Strides(*looks)
    )
    from tqdm.auto import tqdm

    for out_idxs, _, in_idxs, _, _ in tqdm(list(block_manager.iter_blocks())):
        in_rows, in_cols = in_idxs
        out_rows, out_cols = out_idxs
        arr_ref = reader_ref[in_rows, in_cols].filled(0)
        arr_sec = reader_sec[in_rows, in_cols].filled(0)

        ifg = _form_ifg(arr_ref, arr_sec, looks)
        io.write_block(
            np.asarray(ifg),
            filename=out_path,
            row_start=out_rows.start,
            col_start=out_cols.start,
        )

    return out_path


def create_cor(ifg_filename: Path | str, outfile: Optional[Path | str] = None) -> Path:
    """Write out a binary correlation file for an interferogram.

    Assumes the interferogram has been normalized so that the absolute value
    is the correlation.

    Parameters
    ----------
    ifg_filename : Path | str
        Complex interferogram filename
    outfile : Optional[Path | str], optional
        Output filename, by default None
        If None, will use the same name as the interferogram but with the
        extension changed to .cor

    Returns
    -------
    Path
        Output filename
    """
    from osgeo_utils import gdal_calc

    if outfile is None:
        outfile = Path(ifg_filename).with_suffix(".cor.tif")
    if Path(outfile).exists():
        return Path(outfile)
    gdal_calc.Calc(
        "abs(A)",
        A=str(ifg_filename),
        outfile=str(outfile),
        NoDataValue=0,
        type="Float32",
        creation_options=["compress=lzw", "tiled=yes"],
    )
    return Path(outfile)


@partial(jit, static_argnums=(1, 2, 3))
def take_looks(image, row_looks, col_looks, average=True):
    # Ensure the image has a channel/batch dimension (assuming grayscale image)
    # Add a (batch, ..., channel) dimensions to make NHWC
    image = image[jnp.newaxis, ..., jnp.newaxis]

    # Create a kernel filled with ones
    # Kernel shape: HWIO (height, width, input_channels, output_channels)
    kernel = jnp.ones((row_looks, col_looks, 1, 1), dtype=image.dtype)

    # With each window, we're jumping over by the same number of pixels
    strides = (row_looks, col_looks)
    result = lax.conv_general_dilated(
        image,
        kernel,
        window_strides=strides,
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )

    # Average if required
    if average:
        result /= row_looks * col_looks

    return result.squeeze()


@partial(jit, static_argnums=(2,))
def _form_ifg(arr1, arr2, looks):
    # Phase cross multiply for numerator
    numer = take_looks(arr1 * arr2.conj(), *looks)

    # Normalize so absolute value is correlation
    pow1 = take_looks(arr1 * arr1.conj(), *looks)
    pow2 = take_looks(arr2 * arr2.conj(), *looks)
    denom = jnp.sqrt(pow1 * pow2) + 1e-6
    return numer / denom


def _form_ifg_name(
    slc1: Path | str,
    slc2: Path | str,
    out_dir: Path | str,
    ext: str = ".tif",
    date_fmt: str = "%Y%m%d",
) -> Path:
    date1 = get_dates(slc1, fmt=date_fmt)[0]
    date2 = get_dates(slc2, fmt=date_fmt)[0]
    ifg_name = f"{date1.strftime(date_fmt)}_{date2.strftime(date_fmt)}{ext}"
    return Path(out_dir) / ifg_name


if __name__ == "__main__":
    import tyro

    tyro.cli(create_interferogram)
