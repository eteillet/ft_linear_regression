import numpy as np

def minmax(x, mini=None, maxi=None):
    """Computes the normalized version of a non-empty numpy.ndarray using the min-max standardization.
    Args:
        x: has to be an numpy.ndarray, a vector.
    Returns:
        x’ as a numpy.ndarray.
    """
    if not isinstance(x, np.ndarray) or not np.size(x) or x.shape[1] != 1:
        return None
    if mini == None:
        mini = min(x)
    if maxi == None:
        maxi = max(x)
    return (x - mini) / (maxi - mini)