import os
import logging
import platform

logger = logging.getLogger(__name__)

def initialize_concurrency(config: dict):
    """
    Sets environment variables and library thread limits based on the provided configuration.
    
    Args:
        config: The complete configuration dictionary containing a 'resources' section.
    """
    resources = config.get("resources", {})
    numba_threads = resources.get("numba_threads", 8)
    dask_threads = resources.get("dask_threads", 4)
    # We use dask_threads or numba_threads as a proxy for BLAS/OpenMP
    # Since they are sequential, we can afford to let them use more threads,
    # but we cap them to avoid over-subscription if other processes are running.
    blas_threads = max(numba_threads, dask_threads)

    # Set OpenMP and MKL threads (Do this BEFORE importing numpy, numba, etc.)
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)
    os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)

    # Set Multiprocessing Start Method
    # 'spawn' is safer on Linux when threads are present, avoiding 'cannot join threads' fork errors.
    if platform.system() != 'Windows':
        import multiprocessing
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Multiprocessing start method set to 'spawn'")
        except Exception as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

    # Set Numba threads via API if it was already imported/initialized
    import numba
    try:
        numba.set_num_threads(numba_threads)
        logger.info(f"Numba thread count set to {numba_threads}")
    except Exception as e:
        logger.debug(f"Could not set Numba threads via API: {e}. Environment variable will be used.")

    logger.info(f"BLAS/OpenMP thread limits set to {blas_threads}")
    
    # Dask specific configuration
    try:
        import dask
        dask.config.set(num_workers=dask_threads)
        logger.info(f"Dask global thread limit set to {dask_threads}")
    except ImportError:
        pass
