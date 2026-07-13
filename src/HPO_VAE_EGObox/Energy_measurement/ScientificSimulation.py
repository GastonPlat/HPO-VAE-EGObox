import pandas as pd
from typing import Callable
class Simulation:
    def __init__(self, problem_name: str=None, parameters: dict={}, func: Callable=None): # Parameters example: {'a': 5, 'b': [40,60], 'c': ndarray}
        self.problem_name = problem_name
        self.parameters = parameters
        self.fun = func

    def __repr__(self)->str:
        return self.problem_name

    def run_simulation(self):
        try:
            func = self.fun
            params = self.parameters
            result = func(**params)
            return result
        except TypeError as err:
            print("Argument mismatch:", err)

    def measure_nrj_simu(self, 
                         measure_power_secs=15, 
                         save_to_file=True, 
                         output_file="emissions.csv", 
                         gpu_ids=None, 
                         experiment_id="1", 
                         experiment_name="experiment1", 
                         tracking_mode="process", 
                         on_csv_write="append", 
                         allow_multiple_runs=True,
                         rapl_include_dram=True, 
                         country_iso_code="FRA", 
                         log_level="error"
                         )->pd.DataFrame:

        kwargs_codecarbon = locals().copy()
        kwargs_codecarbon.pop('self', None)

        import os
        from codecarbon import OfflineEmissionsTracker
        from HPO_VAE_EGObox.Energy_measurement.utils.DataframeHandler import get_df, get_last_line

        base_dir = os.path.dirname(os.path.abspath(__file__)) 
        save_dir = os.path.join(base_dir, "CodeCarbon_measures")
        save_dir = os.path.join(save_dir, self.problem_name)
        os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), save_dir), exist_ok=True)
        
        kwargs_codecarbon['output_dir'] = save_dir

        print(f"codecarbon arguments passed: \n {kwargs_codecarbon}")

        tracker = OfflineEmissionsTracker(**kwargs_codecarbon)
        tracker.start()
        
        print("----------")
        print(f"{self.problem_name} started")
        print("----------")
        self.result = self.run_simulation()
        print("----------")
        print(f"{self.problem_name} ended")
        print("----------")

        tracker.stop()

        path = os.path.join(kwargs_codecarbon['output_dir'], kwargs_codecarbon['output_file'])
        data = get_df(full_path_csv = path, n_header = 0)
        last_row = get_last_line(data) # Assuming the most recent input is located in the last line of the .csv

        # return last_row["energy_consumed"].values.item()
        return last_row

        



