import logging
# Initialize logger for this module
logger = logging.getLogger(__name__)
logger.info("Logger initialized. Starting imports...")
from typing import Sequence
import numpy as np
import pandas as pd
import torch
import os
import time

from HPO_VAE_EGObox.VAE.smallestEllipse import *
from HPO_VAE_EGObox.VAE.materialEncoder import MaterialEncoder
from HPO_VAE_EGObox.VAE.utils.utilFuncs import set_seed

class AshbyVAE:
    def __init__(self,  
                 n_neurons_encoder: Sequence[int] = [250], 
                 n_neurons_decoder: Sequence[int] = [250], 
                 dropout_ratio_encoder: float = 0.5,
                 dropout_ratio_decoder: float = 0.5,
                 seed: int = 42,
                 latentDim: int = 2, 
                 numEpochs: int = 10000, 
                 klFactor: float = 4.5e-5, 
                 learningRate: float = 2e-3, 
                 z1_bound_inf: float = -1.5, 
                 z1_bound_sup: float = 1.5, 
                 n_z1: int = 7, 
                 z2_bound_inf: float = -1.0, 
                 z2_bound_sup: float = 1.0, 
                 n_z2: int = 5, 
                 output_VAE_folder: str = "", 
                 path_data: str = './ashby_from_granta_selection3.xlsx', 
                 train_split: float = 0.7,
                 val_split: float = 0.15,
                 test_split: float = 0.15,
                 savedNet: str = './data/vaeNet.nt'):
        
        os.makedirs(output_VAE_folder, exist_ok=True)
        set_seed(seed)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Save splits and ensure they sum to roughly 1.0
        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        assert abs((train_split + val_split + test_split) - 1.0) < 1e-5, "Splits must sum to 1.0"

        # Preprocess data with strict bounds
        data_tensors, self.dataInfo, self.dataIdentifier, self.splitInfo = self.preprocessData(path_data=path_data)
        self.trainingData, self.validationData, self.testingData = data_tensors
        self.trainInfo, self.valInfo, self.testInfo = self.splitInfo
        
        numMaterialsInTrainingData, numFeatures = self.trainingData.shape
        self.latentDim = latentDim 
        self.numEpochs = numEpochs
        self.klFactor = klFactor
        self.learningRate = learningRate
        self.savedNet = savedNet
        
        vaeSettings = {
            'encoder': {
                'inputDim': numFeatures, 
                'latentDim': self.latentDim, 
                'neural_architecture': n_neurons_encoder, 
                'dropout_ratio': dropout_ratio_encoder
            },
            'decoder': {
                'latentDim': self.latentDim, 
                'outputDim': numFeatures, 
                'neural_architecture': n_neurons_decoder, 
                'dropout_ratio': dropout_ratio_decoder
            }
        }
        
        self.props = ['rho', 'E']
        
        self.materialEncoder = MaterialEncoder(
            self.trainingData, 
            self.validationData, 
            self.dataInfo, 
            self.dataIdentifier, 
            vaeSettings, 
            output_VAE_folder
        )

        self.z1_bound_inf = z1_bound_inf
        self.z1_bound_sup = z1_bound_sup
        self.n_z1 = n_z1
        self.z2_bound_inf = z2_bound_inf
        self.z2_bound_sup = z2_bound_sup
        self.n_z2 = n_z2

    def fit(self):
        """
        Train the network and return the final training loss.
        """
        start = time.perf_counter()
        self.convgHistory = self.materialEncoder.trainAutoencoder(self.numEpochs, self.klFactor, self.learningRate)
        logger.info('Training time : {:.2f}s'.format(time.perf_counter() - start))

        self._print_evaluations()

        # === VAE handles ===
        vae = self.materialEncoder.vaeNet
        self.decoder = vae.decoder

        # === Training latent codes ===
        self.z_train = vae.encoder.z.detach()
        self.latent_dim = self.z_train.shape[1]

        self.z_min = self.z_train.min(dim=0)[0]
        self.z_max = self.z_train.max(dim=0)[0]

        self.z_mean = self.z_train.mean(dim=0, keepdim=True)
        self.z_std  = self.z_train.std(dim=0, keepdim=True) + 1e-6

        return self.convgHistory['train_loss'][-1]
    
    def get_test_reconstruction_errors(self) -> dict:
        """
        Calculates and returns the Mean Squared Error (MSE) for the test dataset.
        
        Returns:
            dict: A dictionary where keys are property names (e.g., 'rho', 'E') 
                  and values are the scalar MSE floats.
                  Returns np.nan for each property if test_split is 0.
        """
        # Initialize dictionary to hold scalar MSE values (defaulting to nan)
        errors = {p: np.nan for p in self.props}

        # Safe guard: Return the nan values immediately if there is no test data
        if self.testInfo.shape[0] == 0:
            return errors
        
        # Decode the testing data
        matProp = self.decodeAll(self.testingData)
        
        # Vectorized calculation of MSE for each property
        for p in self.props:
            idx = self.materialEncoder.dataInfo[p]['idx'] # match the correct index for the property
            
            # Retrieve true values (un-log10 them to match the decoded output)
            true_vals = 10**self.testInfo[:, idx]
            
            # Retrieve reconstructed values and convert them to a NumPy array
            recon_vals = matProp[p]
            if torch.is_tensor(recon_vals):
                recon_vals = recon_vals.detach().cpu().numpy()
            
            # Calculate the Mean Squared Error (MSE)
            mse = np.mean((true_vals - recon_vals) ** 2)
            errors[p] = float(mse)
            
        return errors

    def optimize_materials(self, steps=600, lr=0.05, lambda_prior=0.10, clamp=True, return_path=True, clamp_init=True, tol=1e-6):
        # [Unchanged from your original code]
        self.results = []
        z_vals = np.linspace(self.z1_bound_inf, self.z1_bound_sup, self.n_z1)
        z_vals_2 = np.linspace(self.z2_bound_inf, self.z2_bound_sup, self.n_z2)

        z0_list = [[z1, z2] for z1 in z_vals_2 for z2 in z_vals]
        logger.info(f"Initial z0: \n {z0_list}")
        logger.info("Number of start points:", len(z0_list))
        logger.info(f"Running Ashby optimization (sqrt(E)/rho) with {len(z0_list)} user-defined starts...")
        
        for z0 in z0_list:
            z, rho, E, M, z_path, M_path = self.optimize_ashby_beam(
                return_path=return_path, z0=z0, steps=steps, lr=lr, 
                lambda_prior=lambda_prior, clamp=clamp, clamp_init=clamp_init, tol=tol
            )
            self.results.append({"z0": z0, "z": z[0], "rho": rho, "E": E, "M": M, "z_path": z_path, "M_path": M_path})

        self.best = max(self.results, key=lambda r: r["M"])
        logger.info("\n=== Best material (sqrt(E) / rho) ===")
        logger.info(f"start z0 = {self.best['z0']}")
        logger.info(f"rho = {self.best['rho']:.1f} kg/m³")
        logger.info(f"E   = {self.best['E']:.4f} GPa")
        logger.info(f"M   = {self.best['M']:.6g} (sqrt(E)/rho)")
        return self.best

    def preprocessData(self, path_data: str) -> tuple:
        df = pd.read_excel(path_data)

        self.dataIdentifier = {
            "material": df["material"],
            "family": df["Family"],        
            'classID': df['index']
        }

        n_data = self.dataIdentifier['classID'].shape[0]
        
        # Calculate split indices
        idx_train_end = int(n_data * self.train_split)
        idx_val_end = int(n_data * (self.train_split + self.val_split))

        feature_columns = ['rho', 'E']
        all_info = np.log10(df[feature_columns].to_numpy())
        
        trainInfo = all_info[:idx_train_end]
        valInfo = all_info[idx_train_end:idx_val_end]
        testInfo = all_info[idx_val_end:]

        # WARNING FIX: Normalize EVERYTHING using exclusively the training bounds to prevent data leakage
        dataScaleMax_train = torch.tensor(np.max(trainInfo, axis=0), device=self.device)
        dataScaleMin_train = torch.tensor(np.min(trainInfo, axis=0), device=self.device)

        trainingData = (torch.tensor(trainInfo, device=self.device) - dataScaleMin_train) / (dataScaleMax_train - dataScaleMin_train)
        validationData = (torch.tensor(valInfo, device=self.device) - dataScaleMin_train) / (dataScaleMax_train - dataScaleMin_train)
        testingData = (torch.tensor(testInfo, device=self.device) - dataScaleMin_train) / (dataScaleMax_train - dataScaleMin_train)

        dataInfo = {
            'rho': {'idx': 0, 'scaleMin': dataScaleMin_train[0], 'scaleMax': dataScaleMax_train[0]},
            'E':   {'idx': 1, 'scaleMin': dataScaleMin_train[1], 'scaleMax': dataScaleMax_train[1]},
        }

        data_tensors = (trainingData.float(), validationData.float(), testingData.float())
        split_info = (trainInfo, valInfo, testInfo)

        return data_tensors, dataInfo, self.dataIdentifier, split_info

    def unnormalize(self, val, minval, maxval):
        return 10.**(minval + (maxval - minval) * val)

    def decodeAll(self, data_tensor):
        """Modified to decode an arbitrary tensor (train, val, or test)."""
        vae = self.materialEncoder.vaeNet
        with torch.no_grad():
            vae.eval()
            z = vae.encoder(data_tensor)
            decoded = vae.decoder(z)
        matProp = {}
        for k in self.props:
            idx = self.materialEncoder.dataInfo[k]['idx']
            scaleMax = self.materialEncoder.dataInfo[k]['scaleMax']
            scaleMin = self.materialEncoder.dataInfo[k]['scaleMin']
            matProp[k] = self.unnormalize(decoded[:, idx], scaleMin, scaleMax)
        return matProp

    def _print_evaluations(self, display_limit=22):
        """
        Evaluates and prints reconstruction errors for the entire dataset.
        Groups the outputs by Train, Val, and Test splits.
        """
        # Pack splits into a list for easy iteration
        datasets = [
            ('Train', self.trainInfo, self.trainingData),
            ('Val', self.valInfo, self.validationData),
            ('Test', self.testInfo, self.testingData)
        ]

        logger.info('\n----- Per-Material Reconstruction Table -----\n')
        # Added a 'Split' column to the header
        header = f"{'Index':<6} {'Split':<7} {'Family':<20}"
        for p in self.props:
            header += f"{p + ' (True)':>15} {p + ' (Recon)':>15} {p + ' %Err':>10}"
        logger.info(header)
        logger.info('-' * len(header))

        global_idx = 0
        
        # Dictionaries to store maximum and mean errors per split
        maxError = {split: {p: -1e10 for p in self.props} for split, _, _ in datasets}
        meanError = {split: {p: [] for p in self.props} for split, _, _ in datasets}

        # Iterate through Train, Val, and Test sequentially
        for split_name, info_array, data_tensor in datasets:
            # Skip if the split is empty (e.g., if val_split or test_split is 0)
            if info_array.shape[0] == 0:
                continue
                
            # Decode the current dataset split
            matProp = self.decodeAll(data_tensor)
            
            for i in range(info_array.shape[0]):
                family = self.dataIdentifier['material'][global_idx]
                row = f"{global_idx:<6} {split_name:<7} {family:<20}"
                
                for p in self.props:
                    idx = self.materialEncoder.dataInfo[p]['idx']
                    true_val = 10**info_array[i, idx]
                    recon_val = matProp[p][i].item()
                    err_pct = abs(100 * (true_val - recon_val) / true_val)
                    
                    # Store metrics for aggregation
                    meanError[split_name][p].append(err_pct)
                    if err_pct > maxError[split_name][p]:
                        maxError[split_name][p] = err_pct
                    
                    # Format the row if we are under the display limit
                    if display_limit is None or i < display_limit:
                        row += f"{true_val:15.3e} {recon_val:15.3e} {err_pct:10.2f}"
                
                # Print row or omission warning
                if display_limit is None or i < display_limit:
                    logger.info(row)
                elif i == display_limit:
                    logger.info(f"... (remaining {info_array.shape[0] - display_limit} {split_name} items omitted from display) ...")
                    
                global_idx += 1

    def decode_rho_E(self, z):
        decoded = self.decoder(z).squeeze()
        rho = self.unnormalize(decoded[self.dataInfo['rho']['idx']], self.dataInfo['rho']['scaleMin'], self.dataInfo['rho']['scaleMax'])
        E = self.unnormalize(decoded[self.dataInfo['E']['idx']], self.dataInfo['E']['scaleMin'], self.dataInfo['E']['scaleMax'])
        return rho, E, decoded

    def log_ashby_index_beam(self, rho, E, eps=1e-12):
        return 0.5 * torch.log(E + eps) - torch.log(rho + eps)

    def latent_prior(self, z):
        return torch.sum(((z - self.z_mean) / self.z_std) ** 2)

    def optimize_ashby_beam(self, steps=600, lr=0.05, lambda_prior=0.10, clamp=True, return_path=False, z0=None, clamp_init=True, tol=1e-6):
        if z0 is None:
            z0_t = self.z_mean + self.z_std * torch.randn(1, self.latent_dim, device=self.z_train.device, dtype=self.z_train.dtype)
        else:
            z0_t = torch.as_tensor(z0, device=self.z_train.device, dtype=self.z_train.dtype)
            if z0_t.ndim == 1:
                z0_t = z0_t.unsqueeze(0)
            if z0_t.shape != (1, self.latent_dim):
                raise ValueError(f"z0 must have shape (latent_dim,) or (1, latent_dim); got {tuple(z0_t.shape)}")

        if clamp and clamp_init:
            with torch.no_grad():
                z0_t = torch.max(torch.min(z0_t, self.z_max), self.z_min)

        z = torch.nn.Parameter(z0_t.clone())
        optimizer = torch.optim.Adam([z], lr=lr)

        z_path = []
        M_path = []

        if return_path:
            with torch.no_grad():
                rho0, E0, _ = self.decode_rho_E(z)
                M0 = torch.sqrt(E0) / rho0
                z_path.append(z.detach().cpu().numpy().copy()[0])
                M_path.append(float(M0.detach().cpu()))

        for _ in range(steps):
            z_prev = z.detach().clone()
            optimizer.zero_grad()

            rho, E, _ = self.decode_rho_E(z)
            logM = self.log_ashby_index_beam(rho, E)

            loss = -logM + lambda_prior * self.latent_prior(z)
            loss.backward()
            optimizer.step()

            if clamp:
                with torch.no_grad():
                    z.data = torch.max(torch.min(z, self.z_max), self.z_min)

            with torch.no_grad():
                if torch.norm(z - z_prev) < tol:
                    break

            if return_path:
                with torch.no_grad():
                    z_path.append(z.detach().cpu().numpy().copy()[0])
                    M_path.append(float((torch.sqrt(E) / rho).detach().cpu()))

        with torch.no_grad():
            rho, E, _ = self.decode_rho_E(z)
            M = torch.sqrt(E) / rho

        if return_path:
            return (z.detach().cpu().numpy(), float(rho), float(E), float(M), np.asarray(z_path), np.asarray(M_path))
        else:
            return z.detach().cpu().numpy(), float(rho), float(E), float(M)