class DeathError(Exception):
    """
    Raised when all the monkeys in a grid are dead.
    """
    pass

class DataShapeError(Exception):
    """
    Raised when training data does not fit the brain's expected input.
    """
    pass