import csdl_alpha as csdl
import numpy as np
from dataclasses import dataclass
from typing import Union


@dataclass
class ZeroDAirfoilPolarParameters(csdl.VariableGroup):
    """Data class for 1-d airfoil polar.

    Parameters
    ----------
    alpha_stall_plus: float, int
        positive stall angle (deg)
    alpha_stall_minus: float, int
        negative stall angle (deg)
    Cl_stall_plus: float, int
        Cl at positive stall 
    Cl_stall_minus: float, int
        Cl at negative stall 
    Cd_stall_plus: float, int
        Cd at positive stall 
    Cd_stall_minus: float, int
        Cd at negative stall
    Cl_0: float, int
        Cl at zero angle of attack
    Cd_0: float, int
        Cd at zero angle of attack
    Cl_alpha: float, int
        lift-curve slope

    """
    alpha_stall_plus: Union[float, int]
    alpha_stall_minus: Union[float, int]
    Cl_stall_plus: Union[float, int]
    Cl_stall_minus: Union[float, int]
    Cd_stall_plus: Union[float, int]
    Cd_stall_minus: Union[float, int]
    Cl_0: Union[float, int]
    Cd_0: Union[float, int]
    Cl_alpha: Union[float, int]

class ZeroDAirfoilModel:
    def __init__(
            self, 
            polar_parameters: ZeroDAirfoilPolarParameters,
    ) -> None:
        csdl.check_parameter(polar_parameters, "polar_parameters", types=ZeroDAirfoilPolarParameters)

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
    
    def _predict_derivatives(self, AoA_array):
        aoa = AoA_array.flatten()
        cond_list = [
            aoa <= (self.pre_process_parameters.aoa_stall_m-self.pre_process_parameters.eps),
            (aoa > (self.pre_process_parameters.aoa_stall_m-self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_m+self.pre_process_parameters.eps)),
            (aoa > (self.pre_process_parameters.aoa_stall_m+self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_p-self.pre_process_parameters.eps)),
            (aoa > (self.pre_process_parameters.aoa_stall_p-self.pre_process_parameters.eps)) & (aoa <= (self.pre_process_parameters.aoa_stall_p+self.pre_process_parameters.eps)),
            aoa > (self.pre_process_parameters.aoa_stall_p+self.pre_process_parameters.eps)
        ]

        dCl_daoa_fun_list = [
            lambda aoa : 2 * self.pre_process_parameters.A1 * np.cos(2 * aoa) - self.pre_process_parameters.A2_m * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
            lambda aoa : self.pre_process_parameters.coeff_cl_m[2] + 2 * self.pre_process_parameters.coeff_cl_m[1] * aoa + 3 * self.pre_process_parameters.coeff_cl_m[0] * aoa**2,
            lambda aoa : self.pre_process_parameters.Cl_alpha,
            lambda aoa : self.pre_process_parameters.coeff_cl_p[2] + 2 * self.pre_process_parameters.coeff_cl_p[1] * aoa + 3 * self.pre_process_parameters.coeff_cl_p[0] * aoa**2,
            lambda aoa : 2 * self.pre_process_parameters.A1 * np.cos(2 * aoa) - self.pre_process_parameters.A2_p * (np.cos(aoa) * (1+1/(np.sin(aoa))**2)),
        ]

        dCd_daoa_fun_list = [
            lambda aoa : self.pre_process_parameters.B1 * np.sin(2 * aoa) - self.pre_process_parameters.B2_m * np.sin(aoa),
            lambda aoa : self.pre_process_parameters.coeff_cd_m[2] + 2 * self.pre_process_parameters.coeff_cd_m[1] * aoa + 3 * self.pre_process_parameters.coeff_cd_m[0] * aoa**2,
            lambda aoa : 2 * self.pre_process_parameters.k * self.pre_process_parameters.Cl_alpha * (self.pre_process_parameters.Cl_alpha * aoa),
            lambda aoa : self.pre_process_parameters.coeff_cd_p[2] + 2 * self.pre_process_parameters.coeff_cd_p[1] * aoa + 3 * self.pre_process_parameters.coeff_cd_p[0] * aoa**2,
            lambda aoa : self.pre_process_parameters.B1 * np.sin(2 * aoa) - self.pre_process_parameters.B2_p * np.sin(aoa),
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

    def evaluate(self, alpha, Re, Ma):
        
        zero_d_airfoil_model = ZeroDAirfoilCustomOperation(
            airfoil_function=self._predict_values,
            airfoil_function_derivative=self._predict_derivatives,
        )

        return zero_d_airfoil_model.evaluate(alpha)
        

class ZeroDAirfoilCustomOperation(csdl.CustomExplicitOperation):
    def __init__(self, airfoil_function,
                   airfoil_function_derivative):
        self.airfoil_function = airfoil_function
        self.airfoil_function_derivative = airfoil_function_derivative

        super().__init__()

    def evaluate(self, alpha):
        shape = alpha.shape

        self.declare_input("alpha", alpha)
        Cl = self.create_output("Cl", shape)
        Cd = self.create_output("Cd", shape)

        if len(shape) == 2:
            indices = np.arange(shape[0] * shape[1])
        elif len(shape) == 3:
            indices = np.arange(shape[0] * shape[1] * shape[2])
        else: 
            raise NotImplementedError

        self.declare_derivative_parameters("Cl", "alpha", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cd", "alpha", rows=indices, cols=indices)
        
        return Cl, Cd
    
    def compute(self, input_vals, output_vals):
        airfoil_function = self.airfoil_function
        alpha = input_vals["alpha"]
        shape = alpha.shape

        Cl, Cd = airfoil_function(alpha)

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)

    def compute_derivatives(self, inputs, outputs, derivatives):
        airfoil_function_derivative = self.airfoil_function_derivative
        alpha = inputs["alpha"]

        dCl_daoa, dCd_daoa = airfoil_function_derivative(alpha)
        derivatives["Cl", "alpha"] = dCl_daoa
        derivatives["Cd", "alpha"] = dCd_daoa

