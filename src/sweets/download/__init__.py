from ._strategy import DownloadStrategy
from .asf_full import ASFFullDownload
from .burst_subset import BurstSubsetDownload

# Legacy compatibility
ASFQuery = ASFFullDownload

__all__ = ["DownloadStrategy", "ASFFullDownload", "BurstSubsetDownload", "ASFQuery"]
