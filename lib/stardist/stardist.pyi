import numpy as np
from numpy.typing import NDArray

def star_distances(
    src: NDArray[np.uint8], n_rays: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Return star-distance maps for lower and upper bounds."""
