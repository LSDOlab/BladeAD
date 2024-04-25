import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class ZeroDAirfoilPolarParameters(csdl.VariableGroup):
    alpha_stall_plus: Union[float, int]
    alpha_stall_minus: Union[float, int]
    Cl_stall_plus: Union[float, int]
    Cl_stall_minus: Union[float, int]
    Cd_stall_plus: Union[float, int]
    Cd_stall_minus: Union[float, int]
    Cl_0: Union[float, int]
    Cd_0: Union[float, int]
    Cl_alpha: Union[float, int]


class CustomAirfoilModel:
    def __init__(self, **kwargs) -> None:
        self.__dict__.update(kwargs=kwargs)

    def evaluate(self, alpha, Re, Ma):
        raise NotImplementedError
    
    def compute_derivatives(self):
        raise NotImplementedError
    
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
            airfoil_models : list = [],
    ) -> None:
        self.sections = sections
        self.airfoil_models = airfoil_models
        csdl.check_parameter(airfoil_models, "airfoil_models", types=list)
        csdl.check_parameter(sections, "sections", types=list)
        
        if len(sections) < 3:
            raise ValueError("Need at least two sections (i.e., three points) to define multiple airfoils.")
        
        if (len(sections) - 1) != len(airfoil_models):
            raise Exception("length of 'airfoil_models' and (length of 'sections') - 1 must be the same")
        
        if not are_all_elements_of_type(sections, (float, int)):
            raise ValueError("elements of the sections list must either be of type 'float' or 'int'")

        if not is_ascending_between_0_and_1(sections):
            raise ValueError("entries of the sections list must be in ascending order between 0 and 1 order and include the end points.")
        
    def evaluate(self, alpha, Re, Ma):
        shape = alpha.shape
        num_radial = shape[1]

        Cl = csdl.Variable(shape=shape, value=0)
        Cd = csdl.Variable(shape=shape, value=0)

        for i, airfoil_model in enumerate(self.airfoil_models):
            start = self.sections[i]
            stop = self.sections[i+1]

            start_index = int(np.floor(start * num_radial))
            stop_index = int(np.floor(stop * num_radial))

            print(start_index)
            print(stop_index)

            Cl_section, Cd_section = airfoil_model.evaluate(
                alpha[:, start_index:stop_index, :],
                Re[:, start_index:stop_index, :],
                Ma[:, start_index:stop_index, :],
            )

            Cl = Cl.set(csdl.slice[:, start_index:stop_index, :], Cl_section)
            Cd = Cd.set(csdl.slice[:, start_index:stop_index, :], Cd_section)

        return Cl, Cd

class ZeroDAirfoilModel:
    def __init__(
            self, 
            polar_parameters: Union[ZeroDAirfoilPolarParameters, list],
    ) -> None:
        csdl.check_parameter(polar_parameters, "polar_parameters", types=(list, ZeroDAirfoilPolarParameters))

        self.pre_process_parameters = self._process_polar_parameters(polar_parameters)

    def _process_polar_parameters(self, polar_parameters):

        # stall angle > 0
        aoa_stall_p = np.deg2rad(polar_parameters.alpha_stall_plus)

        # Cl at stall > 0 
        Cl_stall_p = polar_parameters.Cl_stall_plus

        # Cd at stall
        Cd_stall_p = polar_parameters.Cd_stall_plus

        # stall angle < 0
        aoa_stall_m = np.deg2rad(polar_parameters.alpha_stall_minus)

        # Cl at stall < 0
        Cl_stall_m = polar_parameters.Cl_stall_minus
        # Cd at stall
        Cd_stall_m = polar_parameters.Cd_stall_minus

        # Cl at zero angle of attack
        Cl_0 = polar_parameters.Cl_0
        
        # Lift curve slope
        Cl_alpha = polar_parameters.Cl_alpha
        
        # Cd at zero angle of attack
        Cd_0 = polar_parameters.Cd_0
        
        # K for quadratic lift polar
        k  = 0.5 * ((Cd_stall_p - Cd_0) / (Cl_0-Cl_stall_p)**2 + (Cd_stall_m - Cd_0) / (Cl_0-Cl_stall_m)**2)
    
        # Smoothing region 
        eps = np.deg2rad(1.5)

        # Viterna Extrapolation 
        AR = 10.
        Cd_max = 1.11 + 0.018 * AR
        A1 = Cd_max / 2
        B1 = Cd_max
        A2_p = (Cl_stall_p - Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * np.sin(aoa_stall_p) / (np.cos(aoa_stall_p)**2)
        A2_m = (Cl_stall_m - Cd_max * np.sin(aoa_stall_m) * np.cos(aoa_stall_m)) * np.sin(aoa_stall_m) / (np.cos(aoa_stall_m)**2)
        B2_p = (Cd_stall_p - Cd_max * np.sin(aoa_stall_p)**2) / np.cos(aoa_stall_p)
        B2_m = (Cd_stall_m - Cd_max * np.sin(aoa_stall_m)**2) / np.cos(aoa_stall_m)

        # Polynomial Smoothing alpha > 0 
        mat_cl_p = mat_cd_p =  np.array([
            [(aoa_stall_p-eps)**3, (aoa_stall_p-eps)**2, (aoa_stall_p-eps), 1],
            [(aoa_stall_p+eps)**3, (aoa_stall_p+eps)**2, (aoa_stall_p+eps), 1],
            [3 * (aoa_stall_p-eps)**2, 2*(aoa_stall_p-eps), 1, 0],
            [3 * (aoa_stall_p+eps)**2, 2*(aoa_stall_p+eps), 1, 0],
        ])

        lhs_cl_p = np.array([
            [Cl_0 + Cl_alpha * (aoa_stall_p-eps)],
            [ A1 * np.sin(2 * (aoa_stall_p+eps)) + A2_p * np.cos(aoa_stall_p+eps)**2 / np.sin(aoa_stall_p+eps)],
            [Cl_alpha],
            [2 * A1 * np.cos(2 * (aoa_stall_p+eps)) - A2_p * (np.cos(aoa_stall_p+eps) * (1+1/(np.sin(aoa_stall_p+eps))**2))],
        ])
        coeff_cl_p = np.linalg.solve(mat_cl_p, lhs_cl_p)

        lhs_cd_p = np.array([
            [Cd_0 + k * (Cl_alpha * (aoa_stall_p-eps))**2],
            [B1 * np.sin(aoa_stall_p+eps)**2 + B2_p * np.cos(aoa_stall_p+eps)],
            [2 * k * Cl_alpha * (Cl_alpha * (aoa_stall_p-eps))],
            [B1 * np.sin(2 * (aoa_stall_p+eps)) - B2_p * np.sin(aoa_stall_p+eps)],
        ])
        coeff_cd_p = np.linalg.solve(mat_cd_p, lhs_cd_p)

        # Polynomial Smoothing alpha < 0 
        mat_cl_m = mat_cd_m =  np.array([
            [(aoa_stall_m-eps)**3, (aoa_stall_m-eps)**2, (aoa_stall_m-eps), 1],
            [(aoa_stall_m+eps)**3, (aoa_stall_m+eps)**2, (aoa_stall_m+eps), 1],
            [3 * (aoa_stall_m-eps)**2, 2*(aoa_stall_m-eps), 1, 0],
            [3 * (aoa_stall_m+eps)**2, 2*(aoa_stall_m+eps), 1, 0],
        ])

        lhs_cl_m = np.array([
            [A1 * np.sin(2 * (aoa_stall_m-eps)) + A2_m * np.cos(aoa_stall_m-eps)**2 / np.sin(aoa_stall_m-eps)],
            [Cl_0 + Cl_alpha * (aoa_stall_m+eps)],
            [2 * A1 * np.cos(2 * (aoa_stall_m-eps)) - A2_m * (np.cos(aoa_stall_m-eps) * (1+1/(np.sin(aoa_stall_m-eps))**2))],
            [Cl_alpha],
        ])
        coeff_cl_m = np.linalg.solve(mat_cl_m, lhs_cl_m)

        lhs_cd_m = np.array([
            [B1 * np.sin(aoa_stall_m-eps)**2 + B2_m * np.cos(aoa_stall_m-eps)],
            [Cd_0 + k * (Cl_alpha * (aoa_stall_m+eps))**2],
            [B1 * np.sin(2 * (aoa_stall_m-eps)) - B2_m * np.sin(aoa_stall_m-eps)],
            [2 * k * Cl_alpha * (Cl_alpha * (aoa_stall_m+eps))],
        ])
        coeff_cd_m = np.linalg.solve(mat_cd_m, lhs_cd_m)

        polar_parameters.eps = eps
        polar_parameters.k = k
        polar_parameters.aoa_stall_m = aoa_stall_m
        polar_parameters.aoa_stall_p = aoa_stall_p
        polar_parameters.A1 = A1
        polar_parameters.B1 = B1
        polar_parameters.A2_p = A2_p
        polar_parameters.A2_m = A2_m
        polar_parameters.B2_p = B2_p
        polar_parameters.B2_m = B2_m
        polar_parameters.coeff_cl_p = coeff_cl_p
        polar_parameters.coeff_cd_p = coeff_cd_p
        polar_parameters.coeff_cl_m = coeff_cl_m
        polar_parameters.coeff_cd_m = coeff_cd_m

        return polar_parameters


    def _predict_values(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.pre_process_parameters.aoa_stall_m-self.pre_process_parameters.eps),
            (aoa > (self.pre_process_parameters.aoa_stall_m-self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_m+self.pre_process_parameters.eps)),
            (aoa > (self.pre_process_parameters.aoa_stall_m+self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_p-self.pre_process_parameters.eps)),
            (aoa > (self.pre_process_parameters.aoa_stall_p-self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_p+self.pre_process_parameters.eps)),
            aoa > (self.pre_process_parameters.aoa_stall_p+self.pre_process_parameters.eps)
        ]
        
        Cl_fun_list = [
            lambda aoa : self.pre_process_parameters.A1 * np.sin(2 * aoa) + self.pre_process_parameters.A2_m * np.cos(aoa)**2 / np.sin(aoa),
            lambda aoa : self.pre_process_parameters.coeff_cl_m[3] + self.pre_process_parameters.coeff_cl_m[2] * aoa + self.pre_process_parameters.coeff_cl_m[1] * aoa**2 + self.pre_process_parameters.coeff_cl_m[0] * aoa**3,
            lambda aoa : self.pre_process_parameters.Cl_0 + self.pre_process_parameters.Cl_alpha * aoa,
            lambda aoa : self.pre_process_parameters.coeff_cl_p[3] + self.pre_process_parameters.coeff_cl_p[2] * aoa + self.pre_process_parameters.coeff_cl_p[1] * aoa**2 + self.pre_process_parameters.coeff_cl_p[0] * aoa**3,
            lambda aoa : self.pre_process_parameters.A1 * np.sin(2 * aoa) + self.pre_process_parameters.A2_p * np.cos(aoa)**2 / np.sin(aoa),
        ]

        Cd_fun_list = [
            lambda aoa : self.pre_process_parameters.B1 * np.sin(aoa)**2 + self.pre_process_parameters.B2_m * np.cos(aoa),
            lambda aoa : self.pre_process_parameters.coeff_cd_m[3] + self.pre_process_parameters.coeff_cd_m[2] * aoa + self.pre_process_parameters.coeff_cd_m[1] * aoa**2 + self.pre_process_parameters.coeff_cd_m[0] * aoa**3,
            lambda aoa : self.pre_process_parameters.Cd_0 + self.pre_process_parameters.k * (self.pre_process_parameters.Cl_alpha * aoa)**2,
            lambda aoa : self.pre_process_parameters.coeff_cd_p[3] + self.pre_process_parameters.coeff_cd_p[2] * aoa + self.pre_process_parameters.coeff_cd_p[1] * aoa**2 + self.pre_process_parameters.coeff_cd_p[0] * aoa**3,
            lambda aoa : self.pre_process_parameters.B1 * np.sin(aoa)**2 + self.pre_process_parameters.B2_p * np.cos(aoa),
        ]    
        
        Cl = np.piecewise(
            aoa, 
            cond_list, 
            Cl_fun_list,
        )

        Cd = np.piecewise(
            aoa, 
            cond_list, 
            Cd_fun_list
        )

        return Cl, Cd

    def evaluate(self, alpha, Re, Ma):
        
        zero_d_airfoil_model = ZeroDAirfoilCustomOperation(
            airfoil_function=self._predict_values
        )

        return zero_d_airfoil_model.evaluate(alpha)
        

class ZeroDAirfoilCustomOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare("airfoil_function")

    def evaluate(self, alpha):
        self.declare_input("alpha", alpha)

        Cl = self.create_output("Cl", alpha.shape)
        Cd = self.create_output("Cd", alpha.shape)
        
        return Cl, Cd
    
    def compute(self, input_vals, output_vals):
        airfoil_function = self.parameters["airfoil_function"]
        alpha = input_vals["alpha"]
        shape = alpha.shape

        Cl, Cd = airfoil_function(alpha)

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)

    def compute_derivatives(self, inputs, outputs, derivatives):
        raise NotImplementedError


class TestAirfoilModel(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare("custom_polar")
    
    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)

        Cl = self.create_output("Cl", alpha.shape)
        Cd = self.create_output("Cd", alpha.shape)

        # TODO: implement derivatives

        return Cl, Cd
    
    def compute(self, input_vals, output_vals):
        custom_polar = ZeroDAirfoilModelImplementaion(self.parameters["custom_polar"])

        alpha = input_vals["alpha"]
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        shape = alpha.shape

        Cl, Cd = custom_polar.predict_values(alpha.flatten())

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)

        

class ZeroDAirfoilModelImplementaion(csdl.CustomExplicitOperation):
    def __init__(self, airfoil_polar: ZeroDAirfoilPolarParameters):
        # stall angle > 0
        self.aoa_stall_p = np.deg2rad(airfoil_polar.alpha_stall_plus)

        # Cl at stall > 0 
        Cl_stall_p = airfoil_polar.Cl_stall_plus

        # Cd at stall
        Cd_stall_p = airfoil_polar.Cd_stall_plus

        # stall angle < 0
        self.aoa_stall_m = np.deg2rad(airfoil_polar.alpha_stall_minus)

        # Cl at stall < 0
        Cl_stall_m = airfoil_polar.Cl_stall_minus
        # Cd at stall
        Cd_stall_m = airfoil_polar.Cd_stall_minus

        # Cl at zero angle of attack
        self.Cl_0 = airfoil_polar.Cl_0
        
        # Lift curve slope
        self.Cl_alpha = airfoil_polar.Cl_alpha
        
        # Cd at zero angle of attack
        self.Cd_0 = airfoil_polar.Cd_0
        
        # K for quadratic lift polar
        self.k  = 0.5 * ((Cd_stall_p - self.Cd_0) / (self.Cl_0-Cl_stall_p)**2 + (Cd_stall_m - self.Cd_0) / (self.Cl_0-Cl_stall_m)**2)
      
        # Smoothing region 
        self.eps = np.deg2rad(1.5)

        # Viterna Extrapolation 
        AR = 10.
        Cd_max = 1.11 + 0.018 * AR
        self.A1 = Cd_max / 2
        self.B1 = Cd_max
        self.A2_p = (Cl_stall_p - Cd_max * np.sin(self.aoa_stall_p) * np.cos(self.aoa_stall_p)) * np.sin(self.aoa_stall_p) / (np.cos(self.aoa_stall_p)**2)
        self.A2_m = (Cl_stall_m - Cd_max * np.sin(self.aoa_stall_m) * np.cos(self.aoa_stall_m)) * np.sin(self.aoa_stall_m) / (np.cos(self.aoa_stall_m)**2)
        self.B2_p = (Cd_stall_p - Cd_max * np.sin(self.aoa_stall_p)**2) / np.cos(self.aoa_stall_p)
        self.B2_m = (Cd_stall_m - Cd_max * np.sin(self.aoa_stall_m)**2) / np.cos(self.aoa_stall_m)

        # Polynomial Smoothing alpha > 0 
        mat_cl_p = mat_cd_p =  np.array([
            [(self.aoa_stall_p-self.eps)**3, (self.aoa_stall_p-self.eps)**2, (self.aoa_stall_p-self.eps), 1],
            [(self.aoa_stall_p+self.eps)**3, (self.aoa_stall_p+self.eps)**2, (self.aoa_stall_p+self.eps), 1],
            [3 * (self.aoa_stall_p-self.eps)**2, 2*(self.aoa_stall_p-self.eps), 1, 0],
            [3 * (self.aoa_stall_p+self.eps)**2, 2*(self.aoa_stall_p+self.eps), 1, 0],
        ])

        lhs_cl_p = np.array([
            [self.Cl_0 + self.Cl_alpha * (self.aoa_stall_p-self.eps)],
            [ self.A1 * np.sin(2 * (self.aoa_stall_p+self.eps)) + self.A2_p * np.cos(self.aoa_stall_p+self.eps)**2 / np.sin(self.aoa_stall_p+self.eps)],
            [self.Cl_alpha],
            [2 * self.A1 * np.cos(2 * (self.aoa_stall_p+self.eps)) - self.A2_p * (np.cos(self.aoa_stall_p+self.eps) * (1+1/(np.sin(self.aoa_stall_p+self.eps))**2))],
        ])
        self.coeff_cl_p = np.linalg.solve(mat_cl_p, lhs_cl_p)

        lhs_cd_p = np.array([
            [self.Cd_0 + self.k * (self.Cl_alpha * (self.aoa_stall_p-self.eps))**2],
            [self.B1 * np.sin(self.aoa_stall_p+self.eps)**2 + self.B2_p * np.cos(self.aoa_stall_p+self.eps)],
            [2 * self.k * self.Cl_alpha * (self.Cl_alpha * (self.aoa_stall_p-self.eps))],
            [self.B1 * np.sin(2 * (self.aoa_stall_p+self.eps)) - self.B2_p * np.sin(self.aoa_stall_p+self.eps)],
        ])
        self.coeff_cd_p = np.linalg.solve(mat_cd_p, lhs_cd_p)

        # Polynomial Smoothing alpha < 0 
        mat_cl_m = mat_cd_m =  np.array([
            [(self.aoa_stall_m-self.eps)**3, (self.aoa_stall_m-self.eps)**2, (self.aoa_stall_m-self.eps), 1],
            [(self.aoa_stall_m+self.eps)**3, (self.aoa_stall_m+self.eps)**2, (self.aoa_stall_m+self.eps), 1],
            [3 * (self.aoa_stall_m-self.eps)**2, 2*(self.aoa_stall_m-self.eps), 1, 0],
            [3 * (self.aoa_stall_m+self.eps)**2, 2*(self.aoa_stall_m+self.eps), 1, 0],
        ])

        lhs_cl_m = np.array([
            [self.A1 * np.sin(2 * (self.aoa_stall_m-self.eps)) + self.A2_m * np.cos(self.aoa_stall_m-self.eps)**2 / np.sin(self.aoa_stall_m-self.eps)],
            [self.Cl_0 + self.Cl_alpha * (self.aoa_stall_m+self.eps)],
            [2 * self.A1 * np.cos(2 * (self.aoa_stall_m-self.eps)) - self.A2_m * (np.cos(self.aoa_stall_m-self.eps) * (1+1/(np.sin(self.aoa_stall_m-self.eps))**2))],
            [self.Cl_alpha],
        ])
        self.coeff_cl_m = np.linalg.solve(mat_cl_m, lhs_cl_m)

        lhs_cd_m = np.array([
            [self.B1 * np.sin(self.aoa_stall_m-self.eps)**2 + self.B2_m * np.cos(self.aoa_stall_m-self.eps)],
            [self.Cd_0 + self.k * (self.Cl_alpha * (self.aoa_stall_m+self.eps))**2],
            [self.B1 * np.sin(2 * (self.aoa_stall_m-self.eps)) - self.B2_m * np.sin(self.aoa_stall_m-self.eps)],
            [2 * self.k * self.Cl_alpha * (self.Cl_alpha * (self.aoa_stall_m+self.eps))],
        ])
        self.coeff_cd_m = np.linalg.solve(mat_cd_m, lhs_cd_m)
    
    def predict_values(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.aoa_stall_m-self.eps),
            (aoa > (self.aoa_stall_m-self.eps)) & (aoa <= (self.aoa_stall_m+self.eps)),
            (aoa > (self.aoa_stall_m+self.eps)) & (aoa <= (self.aoa_stall_p-self.eps)),
            (aoa > (self.aoa_stall_p-self.eps)) & (aoa <= (self.aoa_stall_p+self.eps)),
            aoa > (self.aoa_stall_p+self.eps)
        ]
        
        Cl_fun_list = [
            lambda aoa : self.A1 * np.sin(2 * aoa) + self.A2_m * np.cos(aoa)**2 / np.sin(aoa),
            lambda aoa : self.coeff_cl_m[3] + self.coeff_cl_m[2] * aoa + self.coeff_cl_m[1] * aoa**2 + self.coeff_cl_m[0] * aoa**3,
            lambda aoa : self.Cl_0 + self.Cl_alpha * aoa,
            lambda aoa : self.coeff_cl_p[3] + self.coeff_cl_p[2] * aoa + self.coeff_cl_p[1] * aoa**2 + self.coeff_cl_p[0] * aoa**3,
            lambda aoa : self.A1 * np.sin(2 * aoa) + self.A2_p * np.cos(aoa)**2 / np.sin(aoa),
        ]

        Cd_fun_list = [
            lambda aoa : self.B1 * np.sin(aoa)**2 + self.B2_m * np.cos(aoa),
            lambda aoa : self.coeff_cd_m[3] + self.coeff_cd_m[2] * aoa + self.coeff_cd_m[1] * aoa**2 + self.coeff_cd_m[0] * aoa**3,
            lambda aoa : self.Cd_0 + self.k * (self.Cl_alpha * aoa)**2,
            lambda aoa : self.coeff_cd_p[3] + self.coeff_cd_p[2] * aoa + self.coeff_cd_p[1] * aoa**2 + self.coeff_cd_p[0] * aoa**3,
            lambda aoa : self.B1 * np.sin(aoa)**2 + self.B2_p * np.cos(aoa),
        ]    
        
        Cl = np.piecewise(
            aoa, 
            cond_list, 
            Cl_fun_list,
        )

        Cd = np.piecewise(
            aoa, 
            cond_list, 
            Cd_fun_list
        )

        return Cl, Cd

    def predict_derivatives(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.aoa_stall_m-self.eps),
            (aoa > (self.aoa_stall_m-self.eps)) & (aoa <= (self.aoa_stall_m+self.eps)),
            (aoa > (self.aoa_stall_m+self.eps)) & (aoa <= (self.aoa_stall_p-self.eps)),
            (aoa > (self.aoa_stall_p-self.eps)) & (aoa <= (self.aoa_stall_p+self.eps)),
            aoa > (self.aoa_stall_p+self.eps)
        ]

        dCl_daoa_fun_list = [
            lambda aoa : 2 * self.A1 * np.cos(2 * aoa) - self.A2_m * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
            lambda aoa : self.coeff_cl_m[2] + 2 * self.coeff_cl_m[1] * aoa + 3 * self.coeff_cl_m[0] * aoa**2,
            lambda aoa : self.Cl_alpha,
            lambda aoa : self.coeff_cl_p[2] + 2 * self.coeff_cl_p[1] * aoa + 3 * self.coeff_cl_p[0] * aoa**2,
            lambda aoa : 2 * self.A1 * np.cos(2 * aoa) - self.A2_p * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
        ]

        dCd_daoa_fun_list = [
            lambda aoa : self.B1 * np.sin(2 * aoa) - self.B2_m * np.sin(aoa),
            lambda aoa : self.coeff_cd_m[2] + 2 * self.coeff_cd_m[1] * aoa + 3 * self.coeff_cd_m[0] * aoa**2,
            lambda aoa : 2 * self.k * self.Cl_alpha * (self.Cl_alpha * aoa),
            lambda aoa : self.coeff_cd_p[2] + 2 * self.coeff_cd_p[1] * aoa + 3 * self.coeff_cd_p[0] * aoa**2,
            lambda aoa : self.B1 * np.sin(2 * aoa) - self.B2_p * np.sin(aoa),
        ]    

        dCl_daoa = np.piecewise(
            aoa,
            cond_list,
            dCl_daoa_fun_list,
        )

        dCd_daoa = np.piecewise(
            aoa,
            cond_list,
            dCd_daoa_fun_list,
        )

        return dCl_daoa, dCd_daoa
        