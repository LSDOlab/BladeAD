import csdl_alpha as csdl
import numpy as np


def are_all_elements_of_type(lst: list, data_type):
    return all(isinstance(elem, data_type) for elem in lst)


def is_ascending_between_0_and_1(lst):
    lst.sort()  # Sort the list in ascending order
    if lst[0] != 0 or lst[-1] != 1:
        return False  # If list does not start with 0 or end with 1, return False
    
    for i in range(1, len(lst) - 1):
        if not (0 <= lst[i] <= 1):
            return False  # If any intermediate element is not between 0 and 1, return False
        if lst[i] <= lst[i - 1]:
            return False  # If any intermediate element is not strictly greater than the previous, return False
    
    return True

class CompositeAirfoilModel:
    def __init__(
            self, 
            sections : list = [],
            airfoil_models : list = None,
            smoothing=True,
            transition_window: int = 3,
    ) -> None:
        self.sections = sections
        self.airfoil_models = airfoil_models
        self.smoothing = smoothing
        self.transition_window = transition_window
        csdl.check_parameter(airfoil_models, "airfoil_models", types=list, allow_none=True)
        csdl.check_parameter(sections, "sections", types=list)
        csdl.check_parameter(smoothing, "smoothing", types=bool)
        csdl.check_parameter(transition_window, "transition_window", types=int)

        if len(sections) < 3:
            raise ValueError("Need at least two sections (i.e., three points) to define multiple airfoils.")
        
        if airfoil_models is not None:
            if (len(sections) - 1) != len(airfoil_models):
                raise Exception("length of 'airfoil_models' and (length of 'sections') - 1 must be the same")
        
        if not are_all_elements_of_type(sections, (float, int)):
            raise ValueError("elements of the sections list must either be of type 'float' or 'int'")

        if not is_ascending_between_0_and_1(sections):
            raise ValueError("entries of the sections list must be in ascending order between 0 and 1 order and include the end points.")
        
        
    def evaluate(self, alpha, Re, Ma):
        shape = alpha.shape
        
        if len(shape) == 3:
            num_radial = shape[1]
        elif len(shape) == 2:
            num_radial = shape[0]
        else:
            raise NotImplementedError(f"unknown shape {shape}")

        Cl = csdl.Variable(shape=shape, value=0)
        Cd = csdl.Variable(shape=shape, value=0)

        stop_indices = []
        if self.airfoil_models is not None:
            for i, airfoil_model in enumerate(self.airfoil_models):
                start = self.sections[i]
                stop = self.sections[i+1]

                start_index = int(np.floor(start * num_radial))
                stop_index = int(np.floor(stop * num_radial))

                if stop_index != 0 and stop_index != num_radial:
                    stop_indices.append(stop_index)

                if len(shape) == 3:
                    Cl_section, Cd_section = airfoil_model.evaluate(
                        alpha[:, start_index:stop_index, :],
                        Re[:, start_index:stop_index, :],
                        Ma[:, start_index:stop_index, :],
                    )
                    Cl = Cl.set(csdl.slice[:, start_index:stop_index, :], Cl_section)
                    Cd = Cd.set(csdl.slice[:, start_index:stop_index, :], Cd_section)
                
                elif len(shape) == 2:
                    Cl_section, Cd_section = airfoil_model.evaluate(
                        alpha[start_index:stop_index, :],
                        Re[start_index:stop_index, :],
                        Ma[start_index:stop_index, :],
                    )
                    Cl = Cl.set(csdl.slice[start_index:stop_index, :], Cl_section)
                    Cd = Cd.set(csdl.slice[start_index:stop_index, :], Cd_section)

                else:
                    raise NotImplementedError(f"unkown shape {shape}")
        

        if self.smoothing:
            self.stop_indices = stop_indices
            # for stop_index in stop_indices:
            #     if len(shape) == 3:
            #         for i in csdl.frange(2 * self.transition_window):
            #             index = stop_index + i - self.transition_window
            #             Cl = Cl.set(csdl.slice[:, index, :], csdl.average(Cl[:, index-2:index+2, :], axes=(1, )))
            #             Cd = Cd.set(csdl.slice[:, index, :], csdl.average(Cd[:, index-2:index+2, :], axes=(1, )))

            #     elif len(shape) == 2:
            #         for i in csdl.frange(2 * self.transition_window):
            #             index = stop_index + i - self.transition_window
            #             Cl = Cl.set(csdl.slice[index, :], csdl.average(Cl[index-2:index+2, :], axes=(0, )))
            #             Cd = Cd.set(csdl.slice[index, :], csdl.average(Cd[index-2:index+2, :], axes=(0, )))

        else: 
            self.stop_indices = None

        return Cl, Cd