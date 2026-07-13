import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.info("Logger initialized. Starting imports...")

import os
from smt.design_space import DesignSpace 
from HPO_VAE_EGObox.Energy_measurement.ScientificSimulation import Simulation
import numpy as np
import pandas as pd
import time
import inspect
from typing import Callable


class PbSearchMinNrjAccuracy:
    def __init__(self, problem: Simulation, design_space: DesignSpace, codecarbon_kwargs: dict, budget: float = None, accuracy_min: float = None):
        self.pb = problem
        self.ds = design_space
        self.budget_init = budget
        self.cc_kwargs = codecarbon_kwargs
        self.accuracy_min = accuracy_min

    def __repr__(self)->str:
        return self.pb.problem_name
        
    def is_accuracy_constrained(self):
        return self.accuracy_min is not None
    
    def is_energy_constrained(self):
        return self.budget_init is not None
                
    def _variables_to_kwarg(self, x:np.ndarray)->dict:
        """
        x is a 1-D array
        """
        sig = inspect.signature(self.pb.fun)
        kwargs = {}

        required_params = [
            param for param in sig.parameters.values()
            if param.default == inspect.Parameter.empty
        ]

        x_flat = x.flatten()

        for i, param in enumerate(required_params):
            param_name = param.name
            if self.pb.problem_name == "Mass minimization subject to material selection":
                if param_name == "neural_architecture_encoder":
                    n_layer_encoder, n_neuron_encoder = int(x_flat[10]), int(x_flat[12])
                    neural_architecture = [n_neuron_encoder for _ in range(n_layer_encoder)]
                    kwargs[param_name] = neural_architecture

                elif param_name == "neural_architecture_decoder":
                    n_layer_decoder, n_neuron_decoder = int(x_flat[11]), int(x_flat[13])
                    neural_architecture = [n_neuron_decoder for _ in range(n_layer_decoder)]
                    kwargs[param_name] = neural_architecture

                elif param_name == "lr_material_selection" or param_name == "lr_training" or param_name == "klFactor":
                    kwargs[param_name] = 10**x_flat[i]

                else:
                    kwargs[param_name] = x_flat[i]
            
            elif self.pb.problem_name == "VAE material properties reconstructions":
                if param_name == "neural_architecture_encoder":
                    n_layer_encoder, n_neuron_encoder = int(x_flat[5]), int(x_flat[7])
                    neural_architecture = [n_neuron_encoder for _ in range(n_layer_encoder)]
                    kwargs[param_name] = neural_architecture

                elif param_name == "neural_architecture_decoder":
                    n_layer_decoder, n_neuron_decoder = int(x_flat[6]), int(x_flat[8])
                    neural_architecture = [n_neuron_decoder for _ in range(n_layer_decoder)]
                    kwargs[param_name] = neural_architecture

                elif param_name == "lr_training" or param_name == "klFactor":
                    kwargs[param_name] = 10**x_flat[i]
                
                else:
                    kwargs[param_name] = x_flat[i]


            else:
                kwargs[param_name] = x_flat[i]
        
        logger.info(f"kwargs made from x : = {kwargs}")
        return kwargs

    def set_simulation_params(self, x: np.ndarray)->None:
        self.pb: Simulation
        logger.info(f"x = {x}")
        self.pb.parameters = self._variables_to_kwarg(x=x)
        
    def set_accuracy_func(self, fun:Callable, kwargs: dict = None)->None:
        self.fun_accuracy = fun
        self.kwargs_accuracy = kwargs if kwargs is not None else {}

    def get_accuracy_simu(self):
        try:
            if self.kwargs_accuracy:
                return self.fun_accuracy(**self.kwargs_accuracy)
            else:
                return self.fun_accuracy()
        except Exception as e:
            logger.info(f"Get objective value failed. Error: {e}")

    def set_cost_func(self, fun:Callable, kwargs: dict = None)->None:
        self.fun_cost = fun
        self.kwargs_cost = kwargs

    def get_cost_simu(self):
        try:
            if self.kwargs_cost:
                return self.fun_cost(**self.kwargs_cost)
            else:
                return self.fun_cost()
        except Exception as e:
            logger.info(f"Get objective value failed. Error: {e}")

    def func_HPO(self, x: np.ndarray)->tuple:
        """
        Calculates all objectives to be minimized.
        Handles N-objectives by combining arbitrary length simulation objectives with energy.
        """
        # Set hyperparams and cluster configuration
        self.set_simulation_params(x=x)

        # Measure energy consumption during black-box evaluation
        nrj: float = self.pb.measure_nrj_simu(**self.cc_kwargs)["energy_consumed"].values.item()
        logger.info(f"nrj consumed during black-box evaluation = \n {nrj}")
        
        # Get simulation objectives (N-Dimensional)
        if self.fun_accuracy is not None:
            sim_eval = self.get_accuracy_simu()
            sim_eval = np.atleast_1d(sim_eval)
            sim_objs = -sim_eval # We minimize negative accuracy
        elif self.fun_cost is not None:
            sim_eval = self.get_cost_simu()
            sim_eval = np.atleast_1d(sim_eval)
            sim_objs = sim_eval # We minimize cost directly
        else:
            raise ValueError("Issue in blackbox definition func_HPO: No objective function set using set_cost_func() or set_accuracy_func().")

        # Dynamically build the full objectives array
        objectives = np.concatenate((sim_objs, [nrj]))
        self.n_obj = len(objectives)

        if not self.is_energy_constrained() and not self.is_accuracy_constrained():
            self.history.append({'x': self._variables_to_kwarg(x=x), 'objectives': objectives})
            logger.info(f"history = {pd.DataFrame(self.history)}")
            logger.info(f"objectives = {objectives}")
        else:
            raise NotImplementedError("constrained HPO is not implemented yet")

        fail = False
        self.last_func_eval = objectives, fail
        return self.last_func_eval
    
    def multiobj_optim_egobox(self, n_doe:int, n_iters: int, weights: list = None, obj_mins: list = None, obj_maxs: list = None, seed: int = 42, output_dir: str = ""):
        """
        Runs Bayesian optimization using EgoBOX on a scalarized composite objective.
        Optionally normalizes objectives using provided minimum and maximum bounds before applying weights.
        """
        try:
            import egobox as egx
            logger.info("Imported egobox")
        except (ImportError, ModuleNotFoundError):
            logger.error("egobox library is not installed. Please install it via pip.")
            return None

        if obj_mins is not None and obj_maxs is not None:
            np_mins = -np.array(obj_maxs)
            np_maxs = -np.array(obj_mins)
            
            denom = np_maxs - np_mins
        else:
            np_mins, np_maxs, denom = None, None, None

        def composite_objective(X: np.ndarray) -> np.ndarray:
            y_values = []
            for x_row in X:
                if self.is_energy_constrained() and self.is_accuracy_constrained():
                    eval_res, _ = self.func_grouped_HPO(x=x_row)
                    obj_array = eval_res[:self.n_obj]
                else:
                    eval_res, _ = self.func_HPO(x=x_row)
                    obj_array = eval_res
                
                if np_mins is not None and denom is not None:
                    obj_array = (obj_array - np_mins) / denom
                
                if weights is not None:
                    scalar_val = np.dot(obj_array, weights)
                else:
                    scalar_val = np.sum(obj_array)
                
                y_values.append([scalar_val])
            
            return np.array(y_values)

        xspecs = []
        for var in self.ds.design_variables:
            var_type = var.get_typename()
            
            if var_type == "FloatVariable":
                limits = var.get_limits()
                xspecs.append(egx.XSpec(egx.XType.FLOAT, [float(limits[0]), float(limits[1])]))
                
            elif var_type == "IntegerVariable":
                limits = var.get_limits()
                xspecs.append(egx.XSpec(egx.XType.INT, [int(limits[0]), int(limits[1])]))
                
            elif var_type in ["CategoricalVariable", "OrdinalVariable"]:
                num_choices = len(var.values) if hasattr(var, 'values') else 2
                xspecs.append(egx.XSpec(egx.XType.ORD, [num_choices]))
                
            else:
                logger.warning(f"Unknown variable type {var_type}. Defaulting to FLOAT.")
                limits = var.get_limits()
                xspecs.append(egx.XSpec(egx.XType.FLOAT, [float(limits[0]), float(limits[1])]))

        logger.info("Initializing EgoBOX Egor optimizer for the composite objective...")
        egor = egx.Egor(xspecs, n_doe=n_doe, seed=seed)

        start_time = time.time()
        res = egor.minimize(composite_objective, max_iters=n_iters)
        exec_time = time.time() - start_time
        
        self.result = {
            'best_X': res.result.x_opt,
            'best_F': res.result.y_opt,
            'execution_time': exec_time
        }

        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            save_path = os.path.join(output_dir, f'HPO_EgoBOX_ndoe{n_doe}_niter{n_iters}.npy')
            np.save(save_path, self.result)
            logger.info(f"Results saved to {save_path}")
        
        logger.info(f"EgoBOX optimization completed in {exec_time:.2f} seconds.")
        return self.result