import os
import logging
import platform

logger = logging.getLogger(__name__)

def initialize_concurrency(config: dict):
    """
    Sets environment variables and library thread limits based on the provided configuration.
    """
    resources = config.get("resources", {})
    numba_threads = str(resources.get("numba_threads", 8))
    dask_threads = resources.get("dask_threads", 8)

    # Set Multiprocessing Start Method
    if platform.system() != 'Windows':
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Multiprocessing start method set to 'spawn'")
        except Exception as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

    # Set Numba threads via API
    try:
        import numba
        numba.set_num_threads(int(numba_threads))
        logger.info(f"Numba thread count set to {numba_threads}")
    except Exception as e:
        logger.debug(f"Could not set Numba threads: {e}")
    
    # Dask configuration
    try:
        import dask
        dask.config.set(num_workers=dask_threads)
        logger.info(f"Dask global thread limit set to {dask_threads}")
    except ImportError:
        pass
