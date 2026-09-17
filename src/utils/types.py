from numpy import ndarray, ones

def transform(value, shape) -> ndarray:
    if isinstance(value, ndarray):
        return value
    else:
        return value * ones(shape)
