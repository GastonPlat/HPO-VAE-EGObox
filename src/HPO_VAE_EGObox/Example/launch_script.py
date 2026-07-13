from HPO_VAE_EGObox.Energy_measurement.ScientificSimulation import Simulation
from HPO_VAE_EGObox.VAE.HPO_VAE import VAEWrapper

from smt.design_space import DesignSpace, FloatVariable, IntegerVariable

import os

if __name__ == "__main__":
    # Problem instanciation
    problem = Simulation(
        problem_name="VAE material properties reconstructions",
    )

    # Main parameters of the Bayesian optimization loop
    NDOE = 10
    NITER = 1
    SEED = 42

    # Design space settings
    ds = [
          IntegerVariable(100, 15000),                  # epochs_training, 
          FloatVariable(-5, -3),                        # exponent lr_training,
          FloatVariable(0+10e-3, 1-10e-3),              # dropout_encoder, 
          FloatVariable(0+10e-3, 1-10e-3),              # dropout_decoder, 
          FloatVariable(-5, -1),                        # exponent klFactor, 
          IntegerVariable(1, 5),                        # n_layer_encoder
          IntegerVariable(1, 5),                        # n_layer_decoder
          IntegerVariable(2,512),                       # n_neuron_encoder
          IntegerVariable(2,512),                       # n_neuron_decoder        
    ]
    ds = DesignSpace(design_variables = ds, seed = SEED)

    # CodeCarbon params for energy measurement
    code_carbon_param = {
                            'experiment_name': problem.problem_name,
                            'measure_power_secs': 5, #each 15 (default) seconds hardware power usage will be measured 
                            'tracking_mode': "process", #tries to track the process
                            'log_level': "error", 
                            'output_file': "emissions.csv",
                            'country_iso_code':"FRA",
                            'experiment_id':"1",
                            'save_to_file':True, #Default True
                            'on_csv_write': "append",
                            'allow_multiple_runs':True,
                            'rapl_include_dram':True, 
                            }
                

    # Paths for material data and saving
    base_dir = os.path.join("src", "HPO_VAE_EGObox")
    data_vae = os.path.join(base_dir, "Data", 'AshbyVAE', 'Data_material_ashby', 'ashby_from_granta_selection3.xlsx')
    dir_out = f"EGOBOX_HPO_VAE_material_doe{NDOE}_it{NITER}_seed{SEED}"
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", dir_out)
    os.makedirs(output_dir, exist_ok=True)


    # VAE instanciation
    HPO_VAE_instance = VAEWrapper(
                                    problem=problem,
                                    design_space=ds,
                                    codecarbon_kwargs=code_carbon_param,
                                    HPO_output_dir=output_dir,
                                    training_database_dir = data_vae,
                                    path_saved_net = None,
                                    train_split = 0.7, # To play with
                                    valid_split = 0.15, # To play with
                                    test_split = 0.15, # To play with
                                )

    # Optimization setting
    problem.fun = HPO_VAE_instance.training_VAE_material_mech_chara_inputs # black-box function
    HPO_VAE_instance.set_cost_func(HPO_VAE_instance.MSE_test_data) # setting objective functions 

    # Launching optimization
    res = HPO_VAE_instance.multiobj_optim_egobox(
                                                    n_iters = NITER,
                                                    n_doe = NDOE,
                                                    seed = SEED,
                                                    weights=[.25,.25,.5], # f_obj = .25 * rho_norm + .25 * E_norm + .5 * nrj_norm  # To play with
                                                    obj_mins=[0,0,0], #MSE min reconstruct rho , MSE min reconstruct E, nrj min # To play with
                                                    obj_maxs=[1e8, 1e7, 1], #MSE max reconstruct rho , MSE max reconstruct E, nrj max # To play with
                                                    output_dir=output_dir,
                                                )

    print(f"res = \n {res}")


