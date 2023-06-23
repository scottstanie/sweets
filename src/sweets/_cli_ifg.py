from __future__ import annotations

import argparse
from pathlib import Path

import dask.array as da
import h5py
from dolphin import utils
from dolphin.workflows.config import OPERA_DATASET_NAME

from sweets._log import get_log, log_runtime
from sweets._types import Filename
from sweets.interferogram import create_ifg

logger = get_log(__name__)


def _form_ifg_name(slc1: Filename, slc2: Filename, out_dir: Filename) -> Path:
    """Form the name of the interferogram file.

    Parameters
    ----------
    slc1 : Filename
        First SLC
    slc2 : Filename
        Second SLC
    out_dir : Filename
        Output directory

    Returns
    -------
    Path
        Path to the interferogram file.
    """
    date1 = utils.get_dates(slc1)[0]
    date2 = utils.get_dates(slc2)[0]
    fmt = "%Y%m%d"
    ifg_name = f"{date1.strftime(fmt)}_{date2.strftime(fmt)}.h5"
    return Path(out_dir) / ifg_name


def _get_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--slcs", nargs=2, metavar=("ref_slc_file", "sec_slc_file"), required=True
    ),
    parser.add_argument("--dset", default=OPERA_DATASET_NAME)
    parser.add_argument("-l", "--looks", type=int, nargs=2, default=(1, 1))
    parser.add_argument(
        "-o",
        "--outfile",
    )
    parser.add_argument("--n-workers", type=int, default=4)
    args = parser.parse_args()
    if not args.outfile:
        args.outfile = _form_ifg_name(args.slcs[0], args.slcs[1], ".")
        logger.debug(f"Setting outfile to {args.outfile}")
    return args


@log_runtime
def main():
    """Create one interferogram from two SLCs."""
    args = _get_cli_args()
    with h5py.File(args.slcs[0]) as hf1, h5py.File(args.slcs[1]) as hf2:
        da1 = da.from_array(hf1[args.dset])
        da2 = da.from_array(hf2[args.dset])
        create_ifg(da1, da2, args.looks, outfile=args.outfile)


if __name__ == "__main__":
    main()
