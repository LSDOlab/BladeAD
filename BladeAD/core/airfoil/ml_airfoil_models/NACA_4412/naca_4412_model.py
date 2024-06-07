import csdl_alpha as csdl
import numpy as np
import torch
from torch import nn
from BladeAD import _REPO_ROOT_FOLDER


torch.set_default_dtype(torch.float64)

Cd_reg = nn.Sequential(
    nn.Linear(3, 104),
    nn.GELU(),
    
    nn.Linear(104, 93),
    nn.ReLU(),

    nn.Linear(93, 40),
    nn.SELU(),

    nn.Linear(40, 22),
    nn.ReLU(),

    nn.Linear(22, 50),
    nn.LeakyReLU(),

    nn.Linear(50, 115),
    nn.ReLU(),

    nn.Linear(115, 115),
    nn.GELU(),

    nn.Linear(115, 1), 
)

Cd_reg.load_state_dict(torch.load(_REPO_ROOT_FOLDER / 'core/airfoil/ml_airfoil_models/NACA_4412/Cd_neural_net', map_location=torch.device('cpu')))


Cl_reg = nn.Sequential(
            nn.Linear(3, 82), 
            nn.ReLU(),

            nn.Linear(82, 61),
            nn.ReLU(),

            nn.Linear(61, 121), 
            nn.ReLU(),
            
            nn.Linear(121, 30), 
            nn.ReLU(),

            nn.Linear(30, 87), 
            nn.ReLU(),
            
            nn.Linear(87, 81), 
            nn.ReLU(),
            
            nn.Linear(81, 1), 
)

Cl_reg.load_state_dict(torch.load(_REPO_ROOT_FOLDER / 'core/airfoil/ml_airfoil_models/NACA_4412/Cl_neural_net', map_location=torch.device('cpu')))


class NACA4412MLAirfoilModelCustomOperation(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare("Cl_ml_model")
        self.parameters.declare("Cd_ml_model")

        self.X_max_numpy = np.array([90., 8e6, 0.65])
        self.X_min_numpy = np.array([-90., 1e5, 0.])
    
    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)

        Cl = self.create_output("Cl", alpha.shape)
        Cd = self.create_output("Cd", alpha.shape)
        
        return Cl, Cd
    
    def compute(self, input_vals, output_vals):
        Cl_ml_model = self.parameters["Cl_ml_model"]
        Cd_ml_model = self.parameters["Cd_ml_model"]
        
        alpha = input_vals["alpha"] * 180 / np.pi
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        shape = alpha.shape

        input_tensor = np.zeros((alpha.flatten().shape[0], 3))
        input_tensor[:, 0] = alpha.flatten()
        input_tensor[:, 1] = Re.flatten()
        input_tensor[:, 2] = Ma.flatten()

        scaled_input_tensor = (input_tensor - self.X_min_numpy) \
            / (self.X_max_numpy - self.X_min_numpy)

        input_tensor_torch = torch.Tensor(scaled_input_tensor)

        Cl = Cl_ml_model(input_tensor_torch).detach().numpy()
        Cd = Cd_ml_model(input_tensor_torch).detach().numpy()

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)

    def compute_derivatives(self, inputs, outputs, derivatives):
        raise NotImplementedError


class NACA4412MLAirfoilModel:
    Cl_ml_model = Cl_reg
    Cd_ml_model = Cd_reg

    def evaluate(self, alpha, Re, Ma):
        naca_4412_ml_cusotm_operation = NACA4412MLAirfoilModelCustomOperation(
            Cl_ml_model=self.Cl_ml_model,
            Cd_ml_model=self.Cd_ml_model,
        )

        return naca_4412_ml_cusotm_operation.evaluate(alpha, Re, Ma)
    

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    alfa = csdl.ImplicitVariable(shape=(1, ), value=0.)
    Re = 6e6
    M = 0.14

    naca_airfoil_model = NACA4412MLAirfoilModel()
    
    Cl, _ = naca_airfoil_model.evaluate(alfa, Re, M)

    solver = csdl.nonlinear_solvers.BracketedSearch()
    solver.add_state(alfa, Cl, bracket=(-np.deg2rad(8), np.deg2rad(8)))
    solver.run()

    print(alfa.value * 180 / np.pi)

