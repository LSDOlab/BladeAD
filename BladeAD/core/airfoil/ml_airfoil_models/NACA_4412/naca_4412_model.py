import csdl_alpha as csdl
import numpy as np
import torch
from torch import nn
from BladeAD import _REPO_ROOT_FOLDER
from torch.func import vmap, jacfwd, jacrev


device = torch.device("cpu")
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

Cd_reg.load_state_dict(torch.load(_REPO_ROOT_FOLDER / 'core/airfoil/ml_airfoil_models/NACA_4412/Cd_neural_net', map_location=torch.device("cpu"), weights_only=True)) 


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

Cl_reg.load_state_dict(torch.load(_REPO_ROOT_FOLDER / 'core/airfoil/ml_airfoil_models/NACA_4412/Cl_neural_net', map_location=torch.device("cpu"), weights_only=True))


class NACA4412MLAirfoilModelCustomOperation(csdl.CustomExplicitOperation):
    def __init__(self, Cl_model, Cd_model):
        self.Cl_model = Cl_model
        self.Cd_model = Cd_model

        self.X_max = np.array([90., 8e6, 0.65])
        self.X_min = np.array([-90., 1e5, 0.])

        super().__init__()
    
    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)
        self.declare_input("Re", Re)
        self.declare_input("Ma", Ma)

        if isinstance(alpha, (float, int)):
            alpha = csdl.Variable(shape=(1, ), value=alpha)


        shape = alpha.shape

        if len(shape) == 3:
            indices = np.arange(shape[0] * shape[1] * shape[2])

        elif len(shape) == 2:
            indices = np.arange(shape[0] * shape[1])

        elif len(shape) == 1:
            indices = np.arange(shape[0])

        else:
            raise NotImplementedError

        Cl = self.create_output("Cl", alpha.shape)
        Cd = self.create_output("Cd", alpha.shape)

        self.declare_derivative_parameters("Cl", "alpha", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cl", "Re", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cl", "Ma", rows=indices, cols=indices)

        self.declare_derivative_parameters("Cd", "alpha", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cd", "Re", rows=indices, cols=indices)
        self.declare_derivative_parameters("Cd", "Ma", rows=indices, cols=indices)
        
        return Cl, Cd
    
    def compute(self, input_vals, output_vals):
        Cl_model = self.Cl_model
        Cd_model = self.Cd_model
        
        alpha = input_vals["alpha"] * 180 / np.pi
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        shape = alpha.shape

        input_tensor = np.zeros((alpha.flatten().shape[0], 3))
        input_tensor[:, 0] = alpha.flatten()
        input_tensor[:, 1] = Re.flatten()
        input_tensor[:, 2] = Ma.flatten()

        scaled_input_tensor = (input_tensor - self.X_min) \
            / (self.X_max - self.X_min)

        input_tensor_torch = torch.Tensor(scaled_input_tensor).to(device)

        Cl = Cl_model(input_tensor_torch).cpu().detach().numpy()
        Cd = Cd_model(input_tensor_torch).cpu().detach().numpy()

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)

    def compute_derivatives(self, input_vals, output_vals, derivatives):
        Cl_model = self.Cl_model
        Cd_model = self.Cd_model
        X_min = self.X_min
        X_max = self.X_max

        alpha = input_vals["alpha"] * 180 / np.pi
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        shape = alpha.shape

        if len(shape) == 3:
            indices = np.arange(shape[0] * shape[1] * shape[2])

        elif len(shape) == 2:
            indices = np.arange(shape[0] * shape[1])

        elif len(shape) == 1:
            indices = np.arange(shape[0])

        else:
            raise NotImplementedError
        
        size = len(indices)

        input_tensor = np.zeros((size, 3))
        input_tensor[:, 0] = alpha.flatten()
        input_tensor[:, 1] = Re.flatten()
        input_tensor[:, 2] = Ma.flatten()

        # input_tensor_scaled = (input_tensor - X_min) / (X_max - X_min)
        # input_tensor_torch_scaled = torch.tensor(input_tensor_scaled, dtype=torch.float64).to(device)

        # # Cl jacobian
        # d_Cl_d_scaled_tensor = vmap(jacfwd(Cl_model))(input_tensor_torch_scaled).squeeze().detach().numpy()


        # # Cd jacobian
        # d_Cd_d_scaled_tensor = vmap(jacfwd(Cd_model))(input_tensor_torch_scaled).squeeze().detach().numpy()


        # # Chain rule
        # d_scaled_tensor_d_tensor = (1/(X_max - X_min)).reshape((1, 3))
        # dCl_d_inputs = np.einsum('ik, lk->ik', d_Cl_d_scaled_tensor, d_scaled_tensor_d_tensor)
        # dCd_d_inputs = np.einsum('ik, lk->ik', d_Cd_d_scaled_tensor, d_scaled_tensor_d_tensor)

        # Move tensor operations to PyTorch and GPU where possible
        # input_tensor = torch.zeros((size, 3), device=device, dtype=torch.float64)
        # input_tensor[:, 0] = torch.tensor(alpha.flatten(), device=device, dtype=torch.float64)
        # input_tensor[:, 1] = torch.tensor(Re.flatten(), device=device, dtype=torch.float64)
        # input_tensor[:, 2] = torch.tensor(Ma.flatten(), device=device, dtype=torch.float64)

        # Scaling the input
        X_range_np = X_max - X_min  # Precompute range for reuse
        X_range = torch.tensor(X_range_np, device=device, dtype=torch.float64)
        input_tensor_scaled = (input_tensor - X_min) / X_range

        # Cl jacobian
        d_Cl_d_scaled_tensor = vmap(jacrev(Cl_model))(input_tensor_scaled).squeeze() #.detach()

        # Cd jacobian
        d_Cd_d_scaled_tensor = vmap(jacrev(Cd_model))(input_tensor_scaled).squeeze() #.detach()

        # if d_Cl_d_scaled_tensor.ndim == 3:
        #     d_Cl_d_scaled_tensor = d_Cl_d_scaled_tensor.squeeze()  # Remove extra dimensions if necessary

        # if d_Cd_d_scaled_tensor.ndim == 3:
        #     d_Cd_d_scaled_tensor = d_Cd_d_scaled_tensor.squeeze()

        # Precompute the scaling factor for the chain rule
        d_scaled_tensor_d_tensor = (1 / X_range).view(1, 3)

        # Use torch.einsum for the chain rule instead of numpy.einsum
        dCl_d_inputs_torch = d_Cl_d_scaled_tensor * d_scaled_tensor_d_tensor 
        dCd_d_inputs_torch = d_Cd_d_scaled_tensor * d_scaled_tensor_d_tensor 

        # If necessary, convert back to numpy at the very end
        dCl_d_inputs = dCl_d_inputs_torch.detach().cpu().numpy()
        dCd_d_inputs = dCd_d_inputs_torch.detach().cpu().numpy()

        # print("dCl_d_inputs", dCl_d_inputs)

        # Assign derivatives
        derivatives["Cl", "alpha"] = dCl_d_inputs[:, 0] * 180/np.pi
        derivatives["Cl", "Re"] = dCl_d_inputs[:, 1]
        derivatives["Cl", "Ma"] = dCl_d_inputs[:, 2]

        derivatives["Cd", "alpha"] = dCd_d_inputs[:, 0] * 180/np.pi
        derivatives["Cd", "Re"] = dCd_d_inputs[:, 1]
        derivatives["Cd", "Ma"] = dCd_d_inputs[:, 2]


class NACA4412MLAirfoilModel:
    Cl_model = Cl_reg
    Cd_model = Cd_reg

    def evaluate(self, alpha, Re, Ma):
        naca_4412_ml_cusotm_operation = NACA4412MLAirfoilModelCustomOperation(
            Cl_model=self.Cl_model,
            Cd_model=self.Cd_model,
        )

        return naca_4412_ml_cusotm_operation.evaluate(alpha, Re, Ma)
    

if __name__ == "__main__":
    recorder = csdl.Recorder(inline=True)
    recorder.start()

    alfa = csdl.Variable(shape=(1, ), name="alfa", value=np.deg2rad(0.))
    Re = csdl.Variable(shape=(1, ), name="Re", value=6e6)
    M = csdl.Variable(shape=(1, ), name="M", value=0.14)

    naca_airfoil_model = NACA4412MLAirfoilModel()
    
    Cl, Cd = naca_airfoil_model.evaluate(alfa, Re, M)
    Cl.add_name("Cl")
    Cd.add_name("Cd")

    print(Cl.value)
    print(Cd.value)

    from csdl_alpha.src.operations.derivative.utils import verify_derivatives_inline
    verify_derivatives_inline([Cl, Cd], [alfa, Re, M], 1e-6, raise_on_error=False)

    exit()

    solver = csdl.nonlinear_solvers.BracketedSearch()
    solver.add_state(alfa, Cl, bracket=(-np.deg2rad(8), np.deg2rad(8)))
    solver.run()

    print(alfa.value * 180 / np.pi)

