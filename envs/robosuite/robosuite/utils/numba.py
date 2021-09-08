"""
Numba utils.
"""
import numba
from envs.robosuite.robosuite.utils import macros


def jit_decorator(func):
    if macros.ENABLE_NUMBA:
        return numba.jit(nopython=True, cache=macros.CACHE_NUMBA)(func)
    return func
