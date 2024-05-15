import os
from BladeAD import _REPO_ROOT_FOLDER
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import torch 
import torch.nn.functional as F
from torch import nn
import optuna
import pickle
import csdl_alpha as csdl


# Run on GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def define_model(trial):
    n_layers = trial.suggest_int("n_layers", 2, 6)
    layers = []
    in_features = 2

    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 5, 100)
        layers.append(nn.Linear(in_features, out_features))
        activation_fun = trial.suggest_categorical(f"activation_fun_{i}", ["ReLU", "GELU", "SELU", "Tanh", "Softplus"])
        layers.append(getattr(nn, activation_fun)())
        in_features = out_features
    layers.append(nn.Linear(in_features, 1))

    return nn.Sequential(*layers)

def train_two_d_airfoil_model(
        airfoil_name: str, 
        force_retrain: bool=False, 
        tune_hyper_parameters: bool=False, 
        num_trials: int=500, 
        plot_model: bool=False
    ):

    data_directory_path =  f"{_REPO_ROOT_FOLDER}/core/airfoil/xfoil/two_d_data/{airfoil_name}"

    # pre-allocate 2-D inputs/outputs
    training_inputs = np.zeros((1, 2))
    training_outputs = np.zeros((1, 2))

    # Loop over all data files
    if plot_model:
        fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(12, 8))
        inputs_list = []
    counter = 0
    for data_file in os.listdir(data_directory_path):
        if not data_file.endswith(".txt"):
            pass
        else:
            # Extract reynolds number
            re_start = data_file.find("il-") + len("il-")
            re_end = data_file.find(".txt")
            Re = float(data_file[re_start:re_end])

            # Extract polar data (aoa, Cl, Cd)
            polar = np.loadtxt(f"{data_directory_path}/{data_file}", skiprows=12)
            aoa = np.deg2rad(polar[:, 0])
            Cl = polar[:, 1]
            Cd = polar[:, 2]

            # Get Cl max/min and corresponding aoa (treat as stall condition)
            Cl_max_index = np.where(Cl==np.max(Cl))[0][0]
            Cl_min_index = np.where(Cl==np.min(Cl))[0][0]

            Cl_prestall = savgol_filter(Cl[Cl_min_index:Cl_max_index+1], window_length=15, polyorder=2)
            Cd_prestall = savgol_filter(Cd[Cl_min_index:Cl_max_index+1], window_length=15, polyorder=2)
            aoa_prestall = aoa[Cl_min_index:Cl_max_index+1]

            # Viterna extrapolation to +/- 90 deg
            aoa_stall_p = aoa_prestall[-1]
            aoa_stall_m = aoa_prestall[0]
            Cl_stall_p = Cl_prestall[-1]
            Cl_stall_m = Cl_prestall[0]
            Cd_stall_p = Cd_prestall[-1]
            Cd_stall_m = Cd_prestall[0]

            AR = 10
            Cd_max = 1.11 + 0.018 * AR
            A1 = Cd_max / 2
            B1 = Cd_max
            A2_p = (Cl_stall_p - Cd_max * np.sin(aoa_stall_p) * np.cos(aoa_stall_p)) * (np.sin(aoa_stall_p)/np.cos(aoa_stall_p)**2)
            A2_m = (Cl_stall_m - Cd_max * np.sin(aoa_stall_m) * np.cos(aoa_stall_m)) * (np.sin(aoa_stall_m)/np.cos(aoa_stall_m)**2)

            B2_p = (Cd_stall_p - Cd_max * np.sin(aoa_stall_p)**2) / (np.cos(aoa_stall_p))
            B2_m = (Cd_stall_m - Cd_max * np.sin(aoa_stall_m)**2) / (np.cos(aoa_stall_m))


            num_post_stall_points = 15
            i_vec = np.arange(0, num_post_stall_points)
            half_cos = 1 - np.cos(i_vec * np.pi / (2 * (num_post_stall_points - 1)))

            aoa_post_stall_plus = aoa_stall_p + np.deg2rad(0.5) - (aoa_stall_p + np.deg2rad(0.5) - np.deg2rad(90)) * half_cos
            aoa_post_stall_minus = np.flip(aoa_stall_m - np.deg2rad(0.5)  - (aoa_stall_m - np.deg2rad(0.5) + np.deg2rad(90)) * half_cos)

            Cl_post_stall_m = A1 * np.sin(2 * aoa_post_stall_minus) + A2_m * np.cos(aoa_post_stall_minus)**2 / np.sin(aoa_post_stall_minus)
            Cl_post_stall_p = A1 * np.sin(2 * aoa_post_stall_plus) + A2_p * np.cos(aoa_post_stall_plus)**2 / np.sin(aoa_post_stall_plus)

            Cd_post_stall_m = B1 * np.sin(aoa_post_stall_minus)**2 +  B2_m * np.cos(aoa_post_stall_minus)
            Cd_post_stall_p = B1 * np.sin(aoa_post_stall_plus)**2 +  B2_p * np.cos(aoa_post_stall_plus)

            aoa_total = np.hstack((aoa_post_stall_minus, aoa_prestall, aoa_post_stall_plus)).flatten()
            Cl_total = np.hstack((Cl_post_stall_m, Cl_prestall, Cl_post_stall_p)).flatten()
            Cd_total = np.hstack((Cd_post_stall_m, Cd_prestall, Cd_post_stall_p)).flatten()

            if plot_model:
                color = plt.cm.tab10(counter)

                axs[0, 0].scatter(np.rad2deg(aoa_total), Cl_total, color=color, s=5, label=f"Re={Re}")
                axs[0, 0].set_xlabel("angle of attack (deg)")
                axs[0, 0].set_ylabel("Cl")
                
                axs[0, 1].scatter(np.rad2deg(aoa_total), Cl_total, color=color, s=5, label=f"Re={Re}")
                axs[0, 1].set_xlabel("angle of attack (deg)")
                axs[0, 1].set_ylabel("Cl")
                axs[0, 1].set_xlim([-15, 20])

                axs[1, 0].scatter(np.rad2deg(aoa_total), Cd_total, color=color, s=5, label=f"Re={Re}")
                axs[1, 0].set_xlabel("angle of attack (deg)")
                axs[1, 0].set_ylabel("Cd")

                axs[1, 1].scatter(np.rad2deg(aoa_total), Cd_total, color=color, s=5, label=f"Re={Re}")
                axs[1, 1].set_xlabel("angle of attack (deg)")
                axs[1, 1].set_ylabel("Cd")
                axs[1, 1].set_ylim([0, 0.05])
                axs[1, 1].set_ylim([-15, 20])

            # assemble training data
            inputs = np.zeros((len(aoa_total), 2))
            outputs = np.zeros((len(aoa_total), 2))

            inputs[:, 1] = aoa_total
            inputs[:, 0] = Re
            if plot_model:
                inputs_list.append(inputs)

            outputs[:, 0] = Cl_total
            outputs[:, 1] = Cd_total

            training_inputs = np.vstack((training_inputs, inputs))
            training_outputs = np.vstack((training_outputs, outputs))

            counter += 1

    # "Extrapolate" training data so model remains well-defined
    training_inputs = training_inputs[1:, :]
    Cd_outputs = training_outputs[1:, 1]
    Cl_outputs = training_outputs[1:, 0]

    min_re_index = np.where(training_inputs[:, 0] == np.min(training_inputs[:, 0]))[0]
    max_re_index = np.where(training_inputs[:, 0] == np.max(training_inputs[:, 0]))[0]

    min_re = training_inputs[min_re_index, 0]
    min_re_aoa = training_inputs[min_re_index, 1]
    min_re_Cd = Cd_outputs[min_re_index]
    min_re_Cl = Cl_outputs[min_re_index]

    max_re = training_inputs[max_re_index, 0]
    max_re_aoa = training_inputs[max_re_index, 1]
    max_re_Cd = Cd_outputs[max_re_index]
    max_re_Cl = Cl_outputs[max_re_index]

    artificial_lower_bound_inputs = np.zeros((len(min_re), 2))
    artificial_lower_bound_outputs_Cd = min_re_Cd
    artificial_lower_bound_outputs_Cl = min_re_Cl
    artificial_lower_bound_inputs[:, 0] = 0.1 * min_re
    artificial_lower_bound_inputs[:, 1] = min_re_aoa

    artificial_upper_bound_inputs = np.zeros((len(max_re), 2))
    artificial_upper_bound_outputs_Cd = max_re_Cd
    artificial_upper_bound_outputs_Cl = max_re_Cl
    artificial_upper_bound_inputs[:, 0] = 10 * max_re
    artificial_upper_bound_inputs[:, 1] = max_re_aoa

    training_inputs = np.vstack((training_inputs, artificial_lower_bound_inputs))
    Cd_outputs = np.hstack((Cd_outputs, artificial_lower_bound_outputs_Cd))
    Cl_outputs = np.hstack((Cl_outputs, artificial_lower_bound_outputs_Cl))

    training_inputs = np.vstack((training_inputs, artificial_upper_bound_inputs))
    Cd_outputs = np.hstack((Cd_outputs, artificial_upper_bound_outputs_Cd))
    Cl_outputs = np.hstack((Cl_outputs, artificial_upper_bound_outputs_Cl))

    xt = training_inputs
    yt_Cl = Cl_outputs
    yt_Cd = Cd_outputs

    if os.path.isfile(f"{data_directory_path}/Cl_model"):
        if force_retrain:
            print(":::::::::::::::::::::::::::TRAINING Cl MODEL:::::::::::::::::::::::::::")
            Cl_model = get_ml_model(xt, yt_Cl, data_directory_path, "Cl", tune_hyper_parameters, num_trials)
            Cl_model.eval()

            print(":::::::::::::::::::::::::::TRAINING Cd MODEL:::::::::::::::::::::::::::")
            Cd_model = get_ml_model(xt, yt_Cd, data_directory_path, "Cd", tune_hyper_parameters, num_trials)
            Cd_model.eval()
        else:
            if os.path.isfile(f"{data_directory_path}/tuned_Cl_model_params.pickle"):
                with open(f"{data_directory_path}/tuned_Cl_model_params.pickle", "rb") as pickle_file:
                    params_dict_cl = pickle.load(pickle_file)
                with open(f"{data_directory_path}/tuned_Cd_model_params.pickle", "rb") as pickle_file:
                    params_dict_cd = pickle.load(pickle_file)
            else:
                with open(f"{data_directory_path}/../general_Cl_model_params.pickle", "rb") as pickle_file:
                    params_dict_cl = pickle.load(pickle_file)
                with open(f"{data_directory_path}/../general_Cd_model_params.pickle", "rb") as pickle_file:
                    params_dict_cd = pickle.load(pickle_file)

            Cl_model = build_model_from_parameters(params_dict_cl, None, None, None, None, None, None, False)
            Cl_model.eval()
            Cl_model.load_state_dict(torch.load(f"{data_directory_path}/Cl_model", map_location=torch.device("cpu")))
            
            Cd_model = build_model_from_parameters(params_dict_cd, None, None, None, None, None, None, False)
            Cd_model.eval()
            Cd_model.load_state_dict(torch.load(f"{data_directory_path}/Cd_model", map_location=torch.device("cpu")))

    else:
        print(":::::::::::::::::::::::::::TRAINING Cl MODEL:::::::::::::::::::::::::::")
        Cl_model = get_ml_model(xt, yt_Cl, data_directory_path, "Cl", tune_hyper_parameters, num_trials)
        Cl_model.eval()

        print(":::::::::::::::::::::::::::TRAINING Cd MODEL:::::::::::::::::::::::::::")
        Cd_model = get_ml_model(xt, yt_Cd, data_directory_path, "Cd", tune_hyper_parameters, num_trials)
        Cd_model.eval()

    X_max = np.max(training_inputs, axis=0, keepdims=True) 
    X_min = np.min(training_inputs, axis=0, keepdims=True) 
    
    if plot_model:
        for i, inputs in enumerate(inputs_list):
            color = plt.cm.tab10(i)
            
            aoa_total = inputs[:, 1]

            inputs_scaled = (inputs - X_min) / (X_max - X_min)
            inputs_torch = torch.tensor(inputs_scaled, dtype=torch.float32).to(device)

            Cl = Cl_model(inputs_torch).cpu().detach().numpy().flatten()
            Cd = Cd_model(inputs_torch).cpu().detach().numpy().flatten()

            axs[0, 0].plot(np.rad2deg(aoa_total), Cl, color=color)
            axs[0, 0].set_xlabel("angle of attack (deg)")
            axs[0, 0].set_ylabel("Cl")
            
            axs[0, 1].plot(np.rad2deg(aoa_total), Cl, color=color)
            axs[0, 1].set_xlabel("angle of attack (deg)")
            axs[0, 1].set_ylabel("Cl")
            axs[0, 1].set_xlim([-15, 20])

            axs[1, 0].plot(np.rad2deg(aoa_total), Cd, color=color)
            axs[1, 0].set_xlabel("angle of attack (deg)")
            axs[1, 0].set_ylabel("Cd")

            axs[1, 1].plot(np.rad2deg(aoa_total), Cd, color=color)
            axs[1, 1].set_xlabel("angle of attack (deg)")
            axs[1, 1].set_ylabel("Cd")
            axs[1, 1].set_ylim([0, 0.05])
            axs[1, 1].set_xlim([-15, 20])

        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{data_directory_path}/airfoil_model_plot.png')
        plt.show()
            

    return Cl_model, Cd_model, X_max, X_min

def get_ml_model(input_data, output_data, data_directory_path, type_="Cl", tune_hyper_parameters=False, num_trials=500):
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(input_data, output_data, train_size=0.70, shuffle=True)

    # Normalizing data
    X_max = np.max(X_train_raw, axis=0, keepdims=True) 
    X_min = np.min(X_train_raw, axis=0, keepdims=True) 

    X_train = (X_train_raw - X_min) / (X_max - X_min)
    X_test = (X_test_raw - X_min) / (X_max - X_min)


    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test = torch.tensor(y_test, dtype=torch.float32).reshape(-1, 1).to(device)

    def get_data(batch_size):
        train_loader = torch.utils.data.DataLoader(
            list(zip(X_train, y_train)),
            batch_size=batch_size,
            shuffle=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            list(zip(X_test, y_test)),
            batch_size=batch_size,
            shuffle=True,
        )

        return train_loader, valid_loader

    if tune_hyper_parameters:
        # define objective function for optuna hyper parameter tuning
        def objective(trial):
            model = define_model(trial).to(device)

            optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "AdamW", "Adagrad"])
            lr = trial.suggest_float("lr", 1e-3, 1e-1, log=True)
            weight_decay = trial.suggest_float("weight_decay", 1e-10, 1e-5, log=True)
            optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)

            gamma = trial.suggest_float("gamma", 0.5, 0.99)
            step_size = trial.suggest_int("step_size", 20, 200)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

            epochs = 1000

            batch_size = trial.suggest_int("batch_size", 5, 20)
            train_loader, valid_loader = get_data(batch_size)

            for epoch in range(epochs):
                model.train()
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.view(data.size(0), -1).to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                
                    if type_ == "Cd":
                        penalty = torch.mean(torch.relu(-output))
                    else:
                        penalty = 0

                    loss = F.mse_loss(output, target) + 10 * penalty
                    loss.backward()
                    optimizer.step()

                # Validation of the model.
                scheduler.step()
                model.eval()
                y_pred = model(X_test)
                mse_test = F.mse_loss(y_pred, y_test)
                trial.report(mse_test, epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

            return mse_test

        print(":::::::::::::::::::::::::::TUNING HYPER PARAMETERS (MAY TAKE UP TO A FEW HOURS):::::::::::::::::::::::::::")
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=num_trials, timeout=12* 3600)

        pruned_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.PRUNED])
        complete_trials = study.get_trials(deepcopy=False, states=[optuna.trial.TrialState.COMPLETE])

        print("Study statistics: ")
        print("  Number of finished trials: ", len(study.trials))
        print("  Number of pruned trials: ", len(pruned_trials))
        print("  Number of complete trials: ", len(complete_trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        with open(f"{data_directory_path}/tuned_{type_}_model_params.pickle", "wb") as pickle_file:
            pickle.dump(trial.params, pickle_file)

        model = build_model_from_parameters(trial.params, X_train, X_test, y_train, y_test, data_directory_path, type_)

    else:
        if os.path.isfile(f"{data_directory_path}/tuned_{type_}_model_params.pickle"):
            with open(f"{data_directory_path}/tuned_{type_}_model_params.pickle", "rb") as pickle_file:
                params_dict = pickle.load(pickle_file)
        else:
            with open(f"{data_directory_path}/../general_{type_}_model_params.pickle", "rb") as pickle_file:
                params_dict = pickle.load(pickle_file)

        model = build_model_from_parameters(params_dict, X_train, X_test, y_train, y_test, data_directory_path, type_)
    
    return model

def build_model_from_parameters(params_dict, X_train, X_test, y_train, y_test, data_directory_path, type_, train=True):
    n_unints = []
    activation_fun = []
    for key, value in params_dict.items():
        if "n_units" in key:
            n_unints.append(value)
        elif "activation_fun" in key:
            activation_fun.append(value)

    n_layers = len(n_unints)
    layers = []
    in_features = 2
    for j in range(n_layers):
        out_features = n_unints[j]
        layers.append(nn.Linear(in_features, out_features))
        activation_fun_name = activation_fun[j]
        layers.append(getattr(nn, activation_fun_name)())
        in_features = out_features

    layers.append(nn.Linear(in_features, 1))

    model = nn.Sequential(*layers).to(device)

    if train:
        criterion = nn.MSELoss()

        # Other hyperparameters
        optimizer_name = params_dict["optimizer"]
        lr = params_dict["lr"]
        weight_decay = params_dict["weight_decay"]
        batch_size = params_dict["batch_size"]
        scheduler_step_size = params_dict["step_size"]
        gamma = params_dict["gamma"]
        
        optimizer = getattr(torch.optim, optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step_size, gamma=gamma)

        input_seq = X_train
        output_seq = y_train

        input_batches = input_seq.unfold(1, 2, 1)
        output_batches = output_seq.unfold(1, 1, 1)

        num_epochs = 1000
        for epoch in range(num_epochs):
            # shuffle batches
            perm = torch.randperm(input_batches.size()[0])
            input_batches = input_batches[perm]
            output_batches = output_batches[perm]
            
            # loop over batches
            for k in range(0, input_batches.size()[0], batch_size):
                # get batch
                input_batch = input_batches[k:k+batch_size]
                output_batch = output_batches[k:k+batch_size]
                
                # zero gradients
                optimizer.zero_grad()
                
                # forward pass
                output_pred = model(input_batch)

                if type_ == "Cd":
                    penalty = torch.mean(torch.relu(-output_pred[:, :, :]))
                else:
                    penalty = 0

                # compute loss
                loss = criterion(output_pred, output_batch) + penalty
                # loss = criterion(output_pred, output_batch) 
                
                # backward pass and optimization
                loss.backward()
                optimizer.step()

            test_loss = criterion(model(X_test), y_test)
            scheduler.step()
            print('Epoch [{}/{}], Train Loss: {:.10e}, Test Loss: {:.10e}'.format(epoch+1, num_epochs, loss.item(), test_loss.item()))

        torch.save(model.state_dict(), f"{data_directory_path}/{type_}_model")

    return model

    

class TwoDMLAirfoilModelCustomOp(csdl.CustomExplicitOperation):
    def initialize(self):
        self.parameters.declare("Cl_model")
        self.parameters.declare("Cd_model")
        self.parameters.declare("X_min")
        self.parameters.declare("X_max")
    
    def evaluate(self, alpha, Re, Ma):
        self.declare_input("alpha", alpha)        
        self.declare_input("Re", Re)        
        self.declare_input("Ma", Ma)        

        Cl = self.create_output("Cl", alpha.shape)
        Cd = self.create_output("Cd", alpha.shape)

        return Cl, Cd

    def compute(self, input_vals, output_vals):
        Cl_model = self.parameters["Cl_model"]
        Cd_model = self.parameters["Cd_model"]
        X_min = self.parameters["X_min"]
        X_max = self.parameters["X_max"]

        alpha = input_vals["alpha"]
        Re = input_vals["Re"]
        Ma = input_vals["Ma"]

        shape = alpha.shape

        input_tensor = np.zeros((alpha.flatten().shape[0], 2))
        input_tensor[:, 0] = Re.flatten()
        input_tensor[:, 1] = alpha.flatten()

        input_tensor_scaled = (input_tensor - X_min) / (X_max - X_min)
        input_tensor_torch = torch.tensor(input_tensor_scaled, dtype=torch.float32).to(device)

        Cl = Cl_model(input_tensor_torch).cpu().detach().numpy()
        Cd = Cd_model(input_tensor_torch).cpu().detach().numpy()

        output_vals["Cl"] = Cl.reshape(shape)
        output_vals["Cd"] = Cd.reshape(shape)


class TwoDMLAirfoilModel:
    """Machine learning-based airfoil model based on XFOIL training data.
    
    Cl and Cd are functions of AoA and Reynolds number with Mach assumed to be 0.
    """
    def __init__(
        self,
        airfoil_name: str, 
        force_retrain: bool = False, 
        plot_model: bool = False,
        tune_hyper_parameters: bool = False,
        num_trials: int = 500, 
    ):
        csdl.check_parameter(airfoil_name, "airfoil_name", types=str)
        csdl.check_parameter(force_retrain, "force_retrain", types=bool)
        csdl.check_parameter(tune_hyper_parameters, "tune_hyper_parameters", types=bool)
        csdl.check_parameter(num_trials, "num_trials", types=int)
        csdl.check_parameter(plot_model, "plot_model", types=bool)
        
        data_directory_path =  f"{_REPO_ROOT_FOLDER}/core/airfoil/xfoil/two_d_data/{airfoil_name}"
        parent_directory_path = os.path.dirname(f"{data_directory_path}")
        
        if not os.path.isdir(data_directory_path):
            available_airfoils = []
            for airfoil in os.listdir(parent_directory_path):
                if "." in airfoil:
                    pass
                else:
                    available_airfoils.append(airfoil)
            raise Exception(f"Unknown airfoil '{airfoil_name}'. Available airfoils: {available_airfoils}")
        
        self._Cl_model, self._Cd_model, self._X_max, self._X_min = \
            train_two_d_airfoil_model(
                airfoil_name=airfoil_name,
                force_retrain=force_retrain,
                tune_hyper_parameters=tune_hyper_parameters,
                num_trials=num_trials,
                plot_model=plot_model,
            )

    def evaluate(self, alpha, Re, Ma):
        ml_custom_operation = TwoDMLAirfoilModelCustomOp(
            Cl_model=self._Cl_model,
            Cd_model=self._Cd_model,
            X_min=self._X_min,
            X_max=self._X_max,
        )

        return ml_custom_operation.evaluate(alpha, Re, Ma)
        

if __name__ == "__main__":
    # Cl_model, Cd_model, _, _ = train_two_d_airfoil_model("naca_4412", force_retrain=False, tune_hyper_parameters=False, num_trials=200, plot_model=True)

    recorder = csdl.Recorder(inline=True)
    recorder.start()

    alfa = csdl.ImplicitVariable(shape=(1, ), value=0.)

    airfoil_model = TwoDMLAirfoilModel(
        "naca_4412"
    )

    Re = 1e6
    M = 0.
    Cl, _ = airfoil_model.evaluate(alfa, Re, M)

    solver = csdl.nonlinear_solvers.BracketedSearch(tolerance=1e-6)
    solver.add_state(alfa, Cl, bracket=(np.deg2rad(-6), np.deg2rad(6)))
    solver.run()

    print(alfa.value * 180 /np.pi)

