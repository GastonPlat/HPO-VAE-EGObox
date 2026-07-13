from HPO_VAE_EGObox.VAE.networks import VariationalAutoencoder
import torch
import numpy as np
from HPO_VAE_EGObox.VAE.utils.utilFuncs import to_np
import os

class MaterialEncoder:
    def __init__(self, trainingData, validationData, dataInfo, dataIdentifier, vaeSettings, folder_output_training):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Encoder initialized on: {self.device}")
        
        self.trainingData = trainingData.to(self.device)
        self.validationData = validationData.to(self.device)
        self.dataInfo = dataInfo
        
        self.dataIdentifier = dataIdentifier
        self.vaeSettings = vaeSettings
        self.vaeNet = VariationalAutoencoder(vaeSettings).to(self.device)
        
        self.output_path_vae_training = os.path.join(os.path.dirname(os.path.abspath(__file__)), folder_output_training, 'VAE_model')
        os.makedirs(self.output_path_vae_training, exist_ok=True)
        self.model_save_path = os.path.join(self.output_path_vae_training, 'vaeTrained.pth')

    def loadAutoencoderFromFile(self):
        self.vaeNet.load_state_dict(torch.load(self.model_save_path, map_location=self.device, weights_only=True))
        self.vaeNet.eval() 

    def trainAutoencoder(self, numEpochs, klFactor, learningRate):
        opt = torch.optim.Adam(self.vaeNet.parameters(), lr=learningRate)

        convgHistory = {
            'train_reconLoss': [], 'train_klLoss': [], 'train_loss': [],
            'val_reconLoss': [], 'val_klLoss': [], 'val_loss': []
        }

        best_val_loss = float('inf')
        
        for epoch in range(numEpochs):
            # --- Training Pass ---
            self.vaeNet.train()
            opt.zero_grad()
            predData = self.vaeNet(self.trainingData)
            batch_size = self.trainingData.shape[0]
            
            klLoss = klFactor * self.vaeNet.encoder.kl / batch_size
            reconLoss = ((self.trainingData - predData)**2).sum() / batch_size
            loss = reconLoss + klLoss
            
            loss.backward()
            opt.step()
            
            convgHistory['train_reconLoss'].append(reconLoss.item())
            convgHistory['train_klLoss'].append((klLoss / klFactor).item())
            convgHistory['train_loss'].append(loss.item())

            # --- Validation Pass ---
            if self.validationData.shape[0] > 0:
              self.vaeNet.eval()
              with torch.no_grad():
                  val_predData = self.vaeNet(self.validationData)
                  val_batch_size = self.validationData.shape[0]
                  
                  val_klLoss = klFactor * self.vaeNet.encoder.kl / val_batch_size
                  val_reconLoss = ((self.validationData - val_predData)**2).sum() / val_batch_size
                  val_loss = val_reconLoss + val_klLoss
                  
                  convgHistory['val_reconLoss'].append(val_reconLoss.item())
                  convgHistory['val_klLoss'].append((val_klLoss / klFactor).item())
                  convgHistory['val_loss'].append(val_loss.item())

              # --- Model Checkpointing (Early Stopping principle) ---
              if val_loss.item() < best_val_loss:
                  best_val_loss = val_loss.item()
                  torch.save(self.vaeNet.state_dict(), self.model_save_path)
            else:
                # If no validation data, just save the model on every epoch (or only at the end)
                torch.save(self.vaeNet.state_dict(), self.model_save_path)

            # --- Logging ---
            if epoch % 500 == 0:
                print('Iter {:d} | Train Loss: {:.2E} | Val Loss: {:.2E}'.format(
                    epoch, loss.item(), val_loss.item()
                ))
        
        # Load the best weights discovered across all epochs
        self.loadAutoencoderFromFile()
        return convgHistory

    def getClosestMaterialFromZ(self, z, numClosest=1):
        self.vaeNet.eval()
        with torch.no_grad():
            # Querying the entire training latent space
            zData = self.vaeNet.encoder(self.trainingData).cpu().numpy()

        target_z = to_np(z)
        dist = np.linalg.norm(zData - target_z, axis=1)
        maxDist = np.max(dist)
        distOrder = np.argsort(dist)
        
        matToUseFromDB = {'material': [], 'confidence': []}

        for i in range(numClosest):
            mat = self.dataIdentifier['material'][distOrder[i]]  # Assuming 'material' column was passed
            matToUseFromDB['material'].append(mat)
            if maxDist > 0:
                confidence = 100. * (1. - (dist[distOrder[i]] / maxDist))
            else:
                confidence = 100.0
            matToUseFromDB['confidence'].append(confidence)
            print(f"Closest material {i} : {mat} , confidence {confidence:.2f}")
            
        return matToUseFromDB