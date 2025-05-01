UINT16_MAX = 65535.0
ONE_MILLION = 1_000_000.0


# TODO: refactor yuma-simulation package to accept hyperparameter values natively
def normalize(value: float, max_value: float) -> float:
    """Normalize a value to the [0,1] range based on a given maximum hyperparameter value."""
    try:
        return value / max_value
    except (TypeError, ZeroDivisionError):
        raise ValueError(f"Cannot normalize value={value} with max_value={max_value}")
