"""Adapter that runs ``dolphin.workflows.displacement`` over a stack of CSLCs.

The job here is small: take the geocoded SLCs that COMPASS produced, build a
:class:`dolphin.workflows.config.DisplacementWorkflow` config from a handful
of sweets-friendly knobs, and call :func:`dolphin.workflows.displacement.run`.

dolphin owns everything from this point on:

- phase linking (sequential estimator over ministacks)
- interferogram network selection
- stitching across bursts
- unwrapping (SNAPHU / SPURT / WHIRLWIND)
- displacement timeseries inversion
- velocity estimation

This module is intentionally a thin shim — do not re-implement any of the
above here.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Optional

from pydantic import BaseModel, Field

from ._log import get_log, log_runtime

if TYPE_CHECKING:
    from dolphin.workflows.displacement import OutputPaths

logger = get_log(__name__)


UnwrapMethod = Literal["snaphu", "spurt", "whirlwind", "phass"]


class DolphinOptions(BaseModel):
    """Sweets-friendly configuration for the dolphin displacement workflow.

    These map straight onto fields of
    :class:`dolphin.workflows.config.DisplacementWorkflow`. Anything not
    exposed here can still be set after the fact by mutating the returned
    config object before :meth:`run`.
    """

    half_window: tuple[int, int] = Field(
        default=(11, 5),
        description=(
            "Half-window (y, x) for phase linking. The default of (11, 5) gives"
            " roughly square windows for the OPERA 10x5 m geocoded posting."
        ),
    )
    strides: tuple[int, int] = Field(
        default=(6, 12),
        description=(
            "Output strides (y, x). With OPERA 10x5 m posting and the default"
            " (6, 12), outputs end up at 60 x 60 m."
        ),
    )
    ministack_size: int = Field(
        default=10,
        ge=2,
        description="Number of SLCs per ministack for the sequential estimator.",
    )
    use_evd: bool = Field(
        default=False,
        description="Use EVD instead of EMI for phase linking.",
    )
    max_bandwidth: int = Field(
        default=4,
        ge=1,
        description="Form nearest-N interferograms (by index) for the network.",
    )
    nearest_n_coherence: int = Field(
        default=3,
        ge=1,
        description="Save N nearest-neighbor multilooked coherence rasters.",
    )
    unwrap_method: UnwrapMethod = Field(
        default="snaphu",
        description="Unwrapping algorithm to invoke through dolphin.",
    )
    n_parallel_unwrap: int = Field(
        default=4,
        ge=1,
        description="Parallel unwrapping jobs (interferograms in flight).",
    )
    snaphu_ntiles: tuple[int, int] = Field(
        default=(1, 1),
        description="SNAPHU tile grid (rows, cols). (1,1) disables tiling.",
    )
    snaphu_tile_overlap: tuple[int, int] = Field(
        default=(200, 200),
        description="SNAPHU tile overlap (rows, cols).",
    )
    snaphu_cost: Literal["defo", "smooth"] = Field(
        default="defo",
        description="SNAPHU statistical cost mode.",
    )
    run_timeseries: bool = Field(
        default=True,
        description="Run dolphin's timeseries inversion + velocity estimation.",
    )
    gpu_enabled: bool = Field(
        default=False,
        description="Enable GPU acceleration if dolphin can find a usable device.",
    )
    threads_per_worker: int = Field(
        default=4,
        ge=1,
        description="Threads per dolphin worker (sets OMP_NUM_THREADS).",
    )
    n_parallel_bursts: int = Field(
        default=1,
        ge=1,
        description=(
            "Number of separate burst stacks dolphin processes in parallel during"
            " phase linking."
        ),
    )
    block_shape: tuple[int, int] = Field(
        default=(512, 512),
        description="Block shape (rows, cols) used for streaming I/O.",
    )


def build_displacement_config(
    cslc_files: list[Path],
    work_directory: Path,
    *,
    options: Optional[DolphinOptions] = None,
    mask_file: Optional[Path] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
):
    """Build a :class:`DisplacementWorkflow` config from sweets options.

    Parameters
    ----------
    cslc_files
        Geocoded SLC files (COMPASS HDF5 outputs).
    work_directory
        Where dolphin will write its scratch and output products.
    options
        Sweets-side knobs. Defaults if None.
    mask_file
        Optional water/validity mask. Convention: 1 = good, 0 = bad, dtype uint8.
    bounds
        Optional crop bounds (left, bottom, right, top) in EPSG:4326.

    Returns
    -------
    dolphin.workflows.config.DisplacementWorkflow
        Configured workflow ready to be passed to :func:`run_displacement`.

    """
    from dolphin.workflows.config import DisplacementWorkflow

    if options is None:
        options = DolphinOptions()

    work_directory = Path(work_directory).resolve()
    work_directory.mkdir(parents=True, exist_ok=True)

    unwrap_options: dict = {
        "unwrap_method": options.unwrap_method,
        "n_parallel_jobs": options.n_parallel_unwrap,
    }
    if options.unwrap_method == "snaphu":
        unwrap_options["snaphu_options"] = {
            "ntiles": list(options.snaphu_ntiles),
            "tile_overlap": list(options.snaphu_tile_overlap),
            "cost": options.snaphu_cost,
        }

    output_options: dict = {
        "strides": {"y": options.strides[0], "x": options.strides[1]},
    }
    if bounds is not None:
        output_options["bounds"] = list(bounds)
        output_options["bounds_epsg"] = 4326

    # Use model_validate so the nested dolphin sub-models accept dicts
    # rather than requiring us to import each one explicitly here.
    # COMPASS CSLCs store the SLC at `/data/VV` (or `/data/HH` etc.). We
    # default to VV since the rest of sweets is co-pol; users with other
    # polarizations can override the field on the returned config before
    # running.
    cfg = DisplacementWorkflow.model_validate(
        {
            "cslc_file_list": [Path(p).resolve() for p in cslc_files],
            "work_directory": work_directory,
            "mask_file": mask_file,
            "input_options": {"subdataset": "/data/VV"},
            "worker_settings": {
                "gpu_enabled": options.gpu_enabled,
                "threads_per_worker": options.threads_per_worker,
                "n_parallel_bursts": options.n_parallel_bursts,
                "block_shape": list(options.block_shape),
            },
            "phase_linking": {
                "ministack_size": options.ministack_size,
                "half_window": {
                    "y": options.half_window[0],
                    "x": options.half_window[1],
                },
                "use_evd": options.use_evd,
                "output_reference_idx": 0,
            },
            "interferogram_network": {
                "max_bandwidth": options.max_bandwidth,
            },
            "unwrap_options": unwrap_options,
            "timeseries_options": {
                "run_inversion": options.run_timeseries,
                "run_velocity": options.run_timeseries,
            },
            "output_options": output_options,
        }
    )
    return cfg


@log_runtime
def run_displacement(
    cslc_files: list[Path],
    work_directory: Path,
    *,
    options: Optional[DolphinOptions] = None,
    mask_file: Optional[Path] = None,
    bounds: Optional[tuple[float, float, float, float]] = None,
    config_yaml: Optional[Path] = None,
) -> "OutputPaths":
    """Build the dolphin config and run the displacement workflow.

    Parameters
    ----------
    cslc_files
        Geocoded SLCs from COMPASS.
    work_directory
        dolphin work / output directory.
    options
        Sweets-side knobs (defaults to :class:`DolphinOptions`).
    mask_file
        Optional water/validity mask.
    bounds
        Optional crop bounds in EPSG:4326.
    config_yaml
        If given, the resolved dolphin config is dumped to this YAML path
        before running, for reproducibility.

    Returns
    -------
    dolphin.workflows.displacement.OutputPaths
        Dataclass of output paths produced by dolphin.

    """
    from dolphin.workflows.displacement import run

    cfg = build_displacement_config(
        cslc_files=cslc_files,
        work_directory=work_directory,
        options=options,
        mask_file=mask_file,
        bounds=bounds,
    )
    if config_yaml is not None:
        cfg.to_yaml(config_yaml)
        logger.info(f"Wrote dolphin config: {config_yaml}")

    logger.info(
        f"Running dolphin displacement on {len(cfg.cslc_file_list)} CSLCs"
        f" in {cfg.work_directory}"
    )
    return run(cfg)
