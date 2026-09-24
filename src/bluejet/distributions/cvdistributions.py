import secrets
import math

def uniform(a: float = 0.0, b: float = 1.0) -> float:
    """Cryptographically secure uniform sample."""
    u = secrets.randbits(53) / (1 << 53)
    return a + (b - a) * u

def exponentialdist(lam: float) -> float:
    y = uniform()
    while y == 0:
        y = uniform()
    return -(1 / lam) * math.log(y)
