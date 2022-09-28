import torch
import torch.nn.functional as F
import torch.utils.data as utils
from torch.autograd import Variable
from torch import nn, optim

import sys
import pickle
import math
import numpy as np
from scipy.io import loadmat, savemat
import os

from Staf_GATE import *


seed = 0
if("--seed" in  sys.argv):
    seed = sys.argv[sys.argv.index("--seed") + 1]
seed = int(seed)

epoch = 5000
if("--epoch" in  sys.argv):
    epoch = sys.argv[sys.argv.index("--epoch") + 1]
epoch = int(epoch)

latent_dim = 68
if("--latent_dim" in  sys.argv):
    latent_dim = sys.argv[sys.argv.index("--latent_dim") + 1] 
latent_dim = int(latent_dim)

lambd = 20
if("--lambd" in  sys.argv):
    lambd = sys.argv[sys.argv.index("--lambd") + 1] 
lambd = int(lambd)

batch_size = 64
if("--batch_size" in  sys.argv):
    batch_size = sys.argv[sys.argv.index("--batch_size") + 1] 
batch_size = int(batch_size)

lr = 0.0001
if("--batch_size" in  sys.argv):
    lr = sys.argv[sys.argv.index("--lr") + 1] 
lr = float(lr)

# Scaling factor for SC
offset = 100 
# Set the neighborhood size for Graph-KNN
n_size = 32
# remove subcortical ROIs
delete_index = list(range(34,49)) + [83]
mat_data = loadmat("../data/example_data.mat")

scs = []
scs_flatten = []
nofill = []

def upper_tri_masking(A):
    m = A.shape[0]
    r = np.arange(m)
    mask = r[:,None] < r
    return A[mask]

for i in range(0, len(mat_data['SC'])):
    temp_sc_mat = np.float32(mat_data['SC'][i,:,:])
    temp_sc_mat = np.delete(temp_sc_mat, delete_index, axis=0)
    temp_sc_mat = np.delete(temp_sc_mat, delete_index, axis=1)

    temp_sc_mat = temp_sc_mat - np.diag(np.diag(temp_sc_mat))
    nofill.append(temp_sc_mat)
    
    np.fill_diagonal(temp_sc_mat, np.mean(temp_sc_mat, 0))
    scs.append(temp_sc_mat)
    scs_flatten.append((temp_sc_mat/offset).flatten())

fcs = []
fcs_flatten = []

for i in range(0, len(mat_data['FC'])):
    temp_fc_mat = np.float32(mat_data['FC'][i,:,:])
    temp_fc_mat = np.delete(temp_fc_mat, delete_index, axis=0)
    temp_fc_mat = np.delete(temp_fc_mat, delete_index, axis=1)

    fcs.append(temp_fc_mat - np.identity(68))
    fcs_flatten.append(upper_tri_masking(fcs[-1]).flatten())

nofill = np.array(nofill)
print("Extracted {} scs from the dataset".format(len(scs)))

A_mat     = -np.mean(nofill, axis=0)
tensor_SC = torch.stack([torch.Tensor(i) for i in scs_flatten])
tensor_FC = torch.stack([torch.Tensor(i) for i in fcs_flatten])

net_data     = utils.TensorDataset(torch.Tensor(tensor_SC), torch.Tensor(tensor_FC))
train_length = int(0.9*len(net_data))
test_length  = len(net_data) - train_length
train_data, test_data = utils.random_split(net_data, [train_length,test_length])
train_loader = utils.DataLoader(train_data, batch_size)
test_loader  = utils.DataLoader(test_data, batch_size)


torch.pi = torch.acos(torch.zeros(1)).item() * 2
def loss_function(recon_x, x, mu, logvar, fc, fc_pred, sig, alph, lambd=20, coef_param=0.5):
    # Poisson NLL Loss for SC reconstruction
    SC_recon     = F.poisson_nll_loss(recon_x, x.reshape(recon_x.shape), reduction='sum', log_input=True)
    # KL divergence for latent parameters
    KLD          = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    normal       = torch.distributions.normal.Normal(torch.zeros(fc.shape).to(device), torch.ones(fc.shape).to(device))
    # calculate skew parameter
    delta        = alph/torch.sqrt((1+alph**2))
    sig_adjusted = sig/(1-(2*delta**2/torch.pi))
    mu_adjusted  = fc - torch.sqrt(sig_adjusted)*delta*torch.sqrt(torch.tensor(2/torch.pi))
    # Skew-Normal Likelihood
    fc_loss      = -torch.sum(normal.log_prob((fc_pred-fc)/torch.sqrt(sig_adjusted)))
    neg_skewness = -torch.sum(torch.log(torch.clip(normal.cdf(alph*(fc_pred-mu_adjusted)/torch.sqrt(sig_adjusted)),min=1e-12)))
    # Regularization
    coef = torch.sum(torch.square(torch.corrcoef(fc_pred) - coef_param))    
    return SC_recon + KLD + fc_loss + neg_skewness + lambd*coef

def train(model, train_loader, lambd=lambd, epoch=epoch, lr=lr):
    model.train()
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    fc_train   = train_loader.dataset.dataset.tensors[1]
    coef_param = 0.2
    loss_list  = []
    train_loss = 0
    
    for epoch in range(epoch):
        train_loss = 0
        for batch_idx, (data) in enumerate(train_loader):
            x = data[0].to(device)
            y = data[1].to(device)
            sig = torch.var(y, dim=0, unbiased=False)
            fc_sig = sig.repeat(len(y), 1).to(device)
            optimizer.zero_grad()
            recon_x, mu, logvar, fc_pred = model(x)
            loss = loss_function(recon_x, x, mu, logvar, y, fc_pred, sig=fc_sig, alph=model.alph,
                lambd=lambd, coef_param=coef_param)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()
        if (epoch+1)%100 == 0:
            print('====> Epoch: {} Average loss: {:.4f}'.format(epoch+1, train_loss / len(train_loader.dataset)))
        loss_list.append(train_loss / len(train_loader.dataset))
    return model, loss_list

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model  = Staf_GATE(latent_dim = latent_dim).to(device)
masks  = []

mask_2NN  = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_2NN)).float())
mask_4NN  = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_4NN)).float())
mask_8NN  = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_8NN)).float())
mask_16NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_16NN)).float())
mask_32NN = (np.argsort(np.argsort(A_mat, axis=-1),axis=-1)<n_size+1)
masks.append(torch.from_numpy(np.float32(mask_32NN)).float())

mask_intercept = np.identity(68*68)
masks.append(torch.from_numpy(np.float32(mask_intercept)).float())
model.set_mask(masks)

print(model)

model, loss_list = train(model=model, train_loader=train_loader, lambd=lambd, epoch=epoch, lr=lr)

if os.path.exists('../result') == False:
    os.mkdir('../result')
output = open('../result/loss.pkl', 'wb')
pickle.dump(loss_list, output)
output.close()

# Extract the latent parameters
num_elements = len(train_loader.dataset)
num_batches  = len(train_loader)
batch_size   = train_loader.batch_size
mu_out       = torch.zeros(num_elements, latent_dim)
logvar_out   = torch.zeros(num_elements,latent_dim)
recon_out    = torch.zeros(num_elements,68*68)

if os.path.exists('../latent_var/') == False:
    os.mkdir('../latent_var/')

with torch.no_grad():
    model.eval()
    for i, (data) in enumerate(train_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data[0].to(device)
        recon_batch, mu, logvar, fc_pred = model(data)
        mu_out[start:end] = mu
        logvar_out[start:end] = logvar
        recon_out[start:end] =recon_batch

savemat('../latent_var/mu_Train_{}.mat'.format(seed), mdict={'mu': mu_out.detach().numpy()})
savemat('../latent_var/logvar_Train_{}.mat'.format(seed), mdict={'logvar': logvar_out.detach().numpy()})

num_elements = len(test_loader.dataset)
num_batches = len(test_loader)
mu_out = torch.zeros(num_elements, latent_dim)
logvar_out = torch.zeros(num_elements,latent_dim)
recon_out = torch.zeros(num_elements,68*68)

with torch.no_grad():
    model.eval()
    for i, (data) in enumerate(test_loader):
        start = i*batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        data = data[0].to(device)
        recon_batch, mu, logvar, fc_pred = model(data)
        mu_out[start:end] = mu
        logvar_out[start:end] = logvar
        recon_out[start:end] =recon_batch

savemat('../latent_var/mu_Test_{}.mat'.format(seed), mdict={'mu': mu_out.detach().numpy()})
savemat('../latent_var/logvar_Test_{}.mat'.format(seed), mdict={'logvar': logvar_out.detach().numpy()})


# Save trained Staf-GATE
torch.save(model.state_dict(), "../result/Staf_GATE_{}.pt".format(seed))


