# HPO-VAE-EGObox
This framework proposes to search for the optimal hyperparameter settings of a Variational AutoEncoder (VAE) using a Bayesian optimization algorithm. 

# How to install
Make sure you already installed python 3.11 or 3.12, and that you are at the root of the repo. Then:
pip install pipx
pipx install poetry 
poetry install

# How to use it:
You will find an example in src/HPO-VAE/Example/launch_script.py. Execute it using poetry run python -m HPO_VAE_EGObox.Example.launch_script for instance.

Globally, the class related to:
- Bayesian optimization is PbSearchMinNrjAccuracy, defined in ProblemHPONrjMin.py.
- energy measurement is Simulation, defined in ScientificSimulation.py
- Variational AutoEncoder (VAE) model is VAEWrapper, defined in HPO_VAE.py. It is a child class of PbSearchMinNrjAccuracy.

# What optimization problem is it solving?
Let's consider the Young modules E, the mass density rho, and the energy consumed by computations nrj
min. weight_1 * normalized_MEAN_SQUARED_ERROR(E_reconstructed, E_true) + weight_2 * normalized_MEAN_SQUARED_ERROR(rho_reconstructed, rho_true) + weight_3 * normalized_nrj

subject to learning rate, dropout_encoder, dropout_decoder, klFactor, # of layer_encoder, # of layer_decoder, # of neuron per layer in encoder, # of neuron per layer in decoder  