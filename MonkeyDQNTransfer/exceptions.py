class DeathError(Exception):
    """
    Raised when all the monkeys in a grid are dead.
    """
    pass

class ControlError(Exception):
	"""
	Raised when the control of the monkeys is not properly defined.
	"""
	pass

class MapSizeError(Exception):
    """
    Raised for non-rectangular maps or inconsistently sized layer maps.
    """
    pass

class SymbolError(Exception):
    """
    Raised for unrecognized symbols.
    """
    pass