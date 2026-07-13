import torch
import torch.nn as nn
import torch.nn.functional as F

#%%
class Encoder(nn.Module):
  def __init__(self, encoderSettings):
    super(Encoder, self).__init__()
    self.layers = nn.ModuleList()
    current_dim = encoderSettings['inputDim']
    
    for u_i in encoderSettings['neural_architecture']: # u_i is the number of neurons in the i-th layer
        self.layers.append(nn.Linear(current_dim, u_i))
        current_dim = u_i

    self.mu_layer = nn.Linear(current_dim, encoderSettings['latentDim'])
    self.logvar_layer = nn.Linear(current_dim, encoderSettings['latentDim'])

    self.dropout = nn.Dropout(p=encoderSettings['dropout_ratio'])

    self.N = torch.distributions.Normal(0, 1)
    self.kl = 0
  def forward(self, x):
    for layer in self.layers:
      x = F.relu(layer(x))
      x = self.dropout(x)
    mu =  self.mu_layer(x)
    logvar = self.logvar_layer(x)
    std = torch.exp(0.5*logvar)
    if(self.training):
      self.z = mu + std*self.N.sample(mu.shape).to(x.device)
    else:
      self.z = mu

    self.kl = -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # the KL formula was not correct
    return self.z
#--------------------------#
class Decoder(nn.Module):
  def __init__(self, decoderSettings):
    super(Decoder, self).__init__()
    self.layers = nn.ModuleList()
    current_dim = decoderSettings['latentDim']
    
    for u_i in decoderSettings['neural_architecture']: # u_i is the number of neurons in the i-th layer
        self.layers.append(nn.Linear(current_dim, u_i))
        current_dim = u_i
      
    self.output_layer = nn.Linear(current_dim, decoderSettings['outputDim'])

    self.dropout = nn.Dropout(p=decoderSettings['dropout_ratio'])

  def forward(self, z):
    for layer in self.layers:
      z = F.relu(layer(z)) #
      z = self.dropout(z)
    z = torch.sigmoid(self.output_layer(z)) # decoder op in range [0,1], change activation function here if considered as a parameter
    return z
#--------------------------#
class VariationalAutoencoder(nn.Module):
  def __init__(self, vaeSettings):
    super(VariationalAutoencoder, self).__init__()

    self.encoder = Encoder(vaeSettings['encoder'])
    self.decoder = Decoder(vaeSettings['decoder'])

  def forward(self, x):
    z = self.encoder(x)
    return self.decoder(z)
#--------------------------#
# #%%
# class MaterialNetwork(nn.Module):
#   def __init__(self, nnSettings):
#     self.nnSettings = nnSettings
#     super().__init__()
#     self.layers = nn.ModuleList()
#     set_seed(1234)
#     current_dim = nnSettings['inputDim']
#     for lyr in range(nnSettings['numLayers']): # define the layers
#       l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
#       nn.init.xavier_normal_(l.weight)
#       nn.init.zeros_(l.bias)
#       self.layers.append(l)
#       current_dim = nnSettings['numNeuronsPerLyr']
#     self.layers.append(nn.Linear(current_dim, nnSettings['outputDim']))
#     self.bnLayer = nn.ModuleList()
#     for lyr in range(nnSettings['numLayers']): # batch norm
#       self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

#   def forward(self, x):
#     m = nn.LeakyReLU();
#     ctr = 0;
#     for layer in self.layers[:-1]: # forward prop
#       x = m(layer(x))#m(self.bnLayer[ctr](layer(x)));
#       ctr += 1;
#     opLayer = self.layers[-1](x)
#     nnOut = torch.sigmoid(opLayer)
#     z = self.nnSettings['zMin'] + self.nnSettings['zRange']*nnOut
#     return z

# #--------------------------#
# #%%
# class TopologyNetwork(nn.Module):
#   def __init__(self, nnSettings):
#     self.inputDim = nnSettings['inputDim']# x and y coordn of the point
#     self.outputDim = nnSettings['outputDim']
#     super().__init__()
#     self.layers = nn.ModuleList()
#     set_seed(1234)
#     current_dim = self.inputDim
#     for lyr in range(nnSettings['numLayers']): # define the layers
#       l = nn.Linear(current_dim, nnSettings['numNeuronsPerLyr'])
#       nn.init.xavier_normal_(l.weight)
#       nn.init.zeros_(l.bias)
#       self.layers.append(l)
#       current_dim = nnSettings['numNeuronsPerLyr']
#     self.layers.append(nn.Linear(current_dim, self.outputDim))
#     self.bnLayer = nn.ModuleList()
#     for lyr in range(nnSettings['numLayers']): # batch norm
#       self.bnLayer.append(nn.BatchNorm1d(nnSettings['numNeuronsPerLyr']))

#   def forward(self, x):
#     m = nn.LeakyReLU()
#     ctr = 0
#     for layer in self.layers[:-1]: # forward prop
#       x = m(self.bnLayer[ctr](layer(x)))
#       ctr += 1
#     opLayer = self.layers[-1](x)
#     rho = torch.sigmoid(opLayer).view(-1)
#     return rho