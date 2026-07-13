# HPO-VAE-EGObox

This framework proposes to search for the optimal hyperparameter settings of a **Variational AutoEncoder (VAE)** using a Bayesian optimization algorithm.

---

## Installation

Ensure you have **Python 3.11** or **3.12** installed and that you are located at the root of the repository. Run the following commands to set up your environment:

```bash
# Install pipx if you haven't already
pip install pipx

# Install poetry using pipx
pipx install poetry 

# Install project dependencies
poetry install

```

---

## Usage

You can find a ready-to-use example in `src/HPO-VAE/Example/launch_script.py`.

To execute the example, run the following command from your terminal:

```bash
poetry run python -m HPO_VAE_EGObox.Example.launch_script

```

---

## Core Architecture

The global architecture is divided into three main components:

| Component | Class Name | File | Description |
| --- | --- | --- | --- |
| **Bayesian Optimization** | `PbSearchMinNrjAccuracy` | `ProblemHPONrjMin.py` | Handles the search for optimal hyperparameters. |
| **Energy Measurement** | `Simulation` | `ScientificSimulation.py` | Manages and evaluates the energy consumption of computations. |
| **VAE Model** | `VAEWrapper` | `HPO_VAE.py` | The Variational AutoEncoder model itself. *(Note: This is a child class of `PbSearchMinNrjAccuracy`)*. |

---

## The Optimization Problem

The framework seeks to minimize a weighted combination of the normalized Mean Squared Error (nMSE) for reconstructed Young's Modulus ($E$) and mass density ($\rho$), alongside the normalized energy consumption ($\text{nrj}$).

### **Objective Function**

$$\min \left( w_1 \cdot \text{nMSE}(E_{\text{recon}}, E_{\text{true}}) + w_2 \cdot \text{nMSE}(\rho_{\text{recon}}, \rho_{\text{true}}) + w_3 \cdot \text{nrj}_{\text{norm}} \right)$$

### **Subject to (Search Space / Hyperparameters)**

The optimization is formulated as the following:

* **Learning rate**
* **Dropout rates** 
* **KL factor** 
* **Network depth** (Number of layers in the encoder and decoder)
* **Network width** (Number of neurons per layer in the encoder and decoder)


 ---

## Data 
Data are located within the folder `src/HPO_VAE_EGObox/Data/AhsbyVAE/Data_material_ashby/ashby_from_granta_selection3.xlsx`. Feel free to use other datasets; however, the framework is based on .xlsx files so be careful. 
