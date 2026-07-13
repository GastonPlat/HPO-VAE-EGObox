from smt.design_space import DesignSpace
from HPO_VAE_EGObox.VAE.Variational_autoencoder_material_selection import AshbyVAE
from HPO_VAE_EGObox.ProblemHPONrjMin import PbSearchMinNrjAccuracy
from HPO_VAE_EGObox.Energy_measurement.ScientificSimulation import Simulation

import os

from typing import Sequence

class VAEWrapper(PbSearchMinNrjAccuracy):
    def __init__(self, 
                 problem: Simulation,
                 design_space: DesignSpace,
                 codecarbon_kwargs: dict,
                 HPO_output_dir:str,
                 training_database_dir:str,
                 path_saved_net:str | None = './data/vaeNet.nt',
                 valid_split:float = 0.15,
                 test_split:float = 0.15,
                 train_split:float = 0.7,
                 ):
        super().__init__(
            problem=problem, 
            design_space=design_space,
            codecarbon_kwargs=codecarbon_kwargs,
        )

        self.output_dir = HPO_output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.training_dir = training_database_dir
        self.savednet_dir = path_saved_net
        self.latent_space_dim = 2

        self.train_split, self.valid_split, self.test_split = train_split, valid_split, test_split

    def __repr__(self):
        return 'Material selection VAE model'
    
    def training_VAE_material_mech_chara_inputs(self,
                       epochs_training: int, 
                       lr_training: float,
                       dropout_encoder: int, 
                       dropout_decoder: int, 
                       klFactor: float, 
                       neural_architecture_encoder: Sequence[int], 
                       neural_architecture_decoder: Sequence[int],
                       seed=42, 
                        )->float:

        # Capture arguments
        trial_params = locals().copy()
        trial_params.pop('self', None) # Remove the class instance from the dictionary

        # Instanciate the VAE
        vae_instance = AshbyVAE(
            n_neurons_encoder = neural_architecture_encoder, 
            n_neurons_decoder = neural_architecture_decoder, 
            dropout_ratio_encoder = dropout_encoder,
            dropout_ratio_decoder = dropout_decoder,
            seed = seed,
            latentDim = self.latent_space_dim, 
            numEpochs = int(epochs_training), 
            klFactor = klFactor, 
            learningRate = lr_training, 
            output_VAE_folder = self.output_dir, 
            path_data = self.training_dir, 
            savedNet = self.savednet_dir,
            train_split = self.train_split,
            val_split = self.valid_split,
            test_split = self.test_split,
        )

        # Train the VAE
        loss_training = vae_instance.fit()

        # Get MSE on reconstructed mechanical properties for the test set (if available)
        self.MSE_test = vae_instance.get_test_reconstruction_errors()

    def MSE_test_data(self)->tuple[float, float]:
        return self.MSE_test['rho'], self.MSE_test['E']