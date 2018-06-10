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

class ControlError(Exception):
	"""
	Raised when the control of the monkeys is not properly defined.
	"""
	pass