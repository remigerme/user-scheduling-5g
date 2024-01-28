from dataclasses import dataclass
import numpy as np

@dataclass
class Solution:
    N: int
    M: int
    K: int
    p: int

    X: np.ndarray
