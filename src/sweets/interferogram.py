#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional
from jax import jit, lax
import jax.numpy as jnp
from functools import partial
import numpy as np
from dolphin import io, Strides
from opera_utils import get_dates

logger = logging.getLogger(__name__)


def create_interferogram(
    ref_slc_file: Path | str,
    sec_slc_file: Path | str,
    outfile: Path | str,
    *,
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
    outfile = Path(outfile)
    if outfile.exists():
        if not overwrite:
            logger.debug(f"Skipping {outfile} because it already exists.")
            return outfile
        else:
            logger.info(f"Overwriting {outfile} because overwrite=True.")
            outfile.unlink()

    gdal_ref_slc_file = io.format_nc_filename(ref_slc_file, dset)
    gdal_sec_slc_file = io.format_nc_filename(sec_slc_file, dset)
    crs = io.get_raster_crs(gdal_ref_slc_file)
    assert crs == io.get_raster_crs(gdal_sec_slc_file)

    logger.info(f"Creating {looks[0]}x{looks[1]} multi-looked interferogram: {outfile}")
    reader_ref = io.HDF5Reader(filename=ref_slc_file, dset_name=dset, nodata=np.nan)
    reader_sec = io.HDF5Reader(filename=sec_slc_file, dset_name=dset, nodata=np.nan)

    io.write_arr(
        arr=None,
        output_name=outfile,
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
            filename=outfile,
            row_start=out_rows.start,
            col_start=out_cols.start,
        )

    return outfile


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


def main():
    """Parse arguments and create an interferogram."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ref-slc-file", type=Path, required=True, help="Path to reference SLC file."
    )
    parser.add_argument(
        "--sec-slc-file", type=Path, required=True, help="Path to secondary SLC file."
    )
    parser.add_argument(
        "--dset", help="For HDF5 SLCs, the dataset to the complex raster"
    )
    parser.add_argument("--outfile", type=Path, help="Output GeoTIFF file.")
    parser.add_argument(
        "--looks", type=int, nargs=2, default=(6, 12), help="Row and column looks."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output."
    )

    args = parser.parse_args()
    if not args.outfile:
        args.outfile = _form_ifg_name(args.ref_slc_file, args.sec_slc_file, ".")
        logger.debug(f"Setting outfile to {args.outfile}")
    create_interferogram(
        ref_slc_file=args.ref_slc_file,
        sec_slc_file=args.sec_slc_file,
        outfile=args.outfile,
        dset=args.dset,
        looks=tuple(args.looks),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
