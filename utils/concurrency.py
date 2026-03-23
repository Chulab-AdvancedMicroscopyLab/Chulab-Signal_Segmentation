import os
import logging
import numba
import multiprocessing
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

    # Set Multiprocessing Start Method
    # 'spawn' is safer on Linux when threads are present, avoiding 'cannot join threads' fork errors.
    if platform.system() != 'Windows':
        try:
            multiprocessing.set_start_method('spawn', force=True)
            logger.info("Multiprocessing start method set to 'spawn'")
        except Exception as e:
            logger.warning(f"Could not set multiprocessing start method: {e}")

    # Set Numba threads
    # Note: Numba's thread count can only be set before any JIT functions are called
    # or via numba.set_num_threads().
    try:
        numba.set_num_threads(numba_threads)
        logger.info(f"Numba thread count set to {numba_threads}")
    except Exception as e:
        logger.warning(f"Could not set Numba threads via API: {e}. Ensure this is called early.")
        os.environ["NUMBA_NUM_THREADS"] = str(numba_threads)

    # Set OpenMP and MKL threads
    os.environ["OMP_NUM_THREADS"] = str(blas_threads)
    os.environ["MKL_NUM_THREADS"] = str(blas_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(blas_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(blas_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(blas_threads)

    logger.info(f"BLAS/OpenMP thread limits set to {blas_threads}")
    
    # Dask specific configuration
    import dask
    dask.config.set(num_workers=dask_threads)
    logger.info(f"Dask global thread limit set to {dask_threads}")
