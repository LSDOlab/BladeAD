import csdl_alpha as csdl
from typing import List


def smooth_quantities_spanwise(
        qty: csdl.Variable, 
        stop_indices: List[int], 
        transition_window: int) -> csdl.Variable:
    """Smooth a radially distributed quantity if it is discontinuous.

    This function is meant to be used in combination with the 
    CompositeAirfoilModel class and smooth the transition between
    airfoil sections. It averages the values across the transition 
    region based on a smoothing window.

    Function call is abstracted from the user.

    Parameters
    ----------
    qty : csdl.Variable
        the quantitiy to be smoothed
    stop_indices : List[int]
        list of blade element indices that specify transition
    transition_window : int
        the length of the smoothing window
        

    Returns
    -------
    csdl.Variable
        the smoothed quantity
    """

    shape = qty.shape

    for stop_index in stop_indices:
        if len(shape) == 3:
            for i in csdl.frange(2 * transition_window):
                index = stop_index + i - transition_window
                qty = qty.set(csdl.slice[:, index, :], csdl.average(qty[:, index-2:index+2, :], axes=(1, )))

        elif len(shape) == 2:
            for i in csdl.frange(2 * transition_window):
                index = stop_index + i - transition_window
                qty = qty.set(csdl.slice[index, :], csdl.average(qty[index-2:index+2, :], axes=(0, )))

    return qty