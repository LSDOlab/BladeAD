import csdl_alpha as csdl
import numpy as np


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
        elif len(shape) == 1:
            indices = np.arange(shape[0])
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
