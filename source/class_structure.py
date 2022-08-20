import torch
import torch.nn.functional as F
from torch.autograd import Variable

def to_var(x, requires_grad=False, volatile=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad, volatile=volatile)

class GraphCNN(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphCNN, self).__init__(in_features, out_features, bias)
        self.mask_flag = False
    def set_mask(self, mask):
        self.mask = to_var(mask, requires_grad=False)
        self.weight.data = self.weight.data*self.mask.data
        self.mask_flag = True 
    def get_mask(self):
        return self.mask
    def forward(self, x):
        if self.mask_flag == True:
            weight = self.weight*self.mask
            return F.linear(x, weight, self.bias)
        else:
            return F.linear(x, self.weight, self.bias)

class Staf_GATE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE_Graph, self).__init__()
        self.fc11 =  nn.Linear(68*68, 1024, bias=False)
        self.fc12 =  nn.Linear(68*68, 1024, bias=False)
        self.fc111 = nn.Linear(1024,128, bias=False)
        self.fc222 = nn.Linear(1024,128, bias=False)
        self.fc21 = nn.Linear(128, latent_dim)
        self.fc22 = nn.Linear(128, latent_dim)

        self.fc3 = nn.Linear(latent_dim, 68)
        self.fc32 = nn.Linear(latent_dim,68)
        self.fc33 = nn.Linear(latent_dim,68)
        self.fc34 = nn.Linear(latent_dim,68)
        self.fc35 = nn.Linear(latent_dim,68)

        self.fc4 = GraphCNN(68,68)
        self.fc5 = GraphCNN(68,68)
        self.fc6 = GraphCNN(68,68)
        self.fc7 = GraphCNN(68,68)
        self.fc8 = GraphCNN(68,68)
        self.fcintercept = GraphCNN(68*68, 68*68)

        self.fc_pred1 = nn.Linear(latent_dim, 68, bias=False)
        self.fc_pred2 = nn.Linear(68, 128, bias=False)
        self.fc_pred3 = nn.Linear(128, 512, bias=False)
        self.fc_pred4 = nn.Linear(512, 1024, bias=False)
        self.fc_pred5 = nn.Linear(1024, 2278, bias=False)
        
        self.alph = nn.Parameter(torch.zeros(2278))

    def encode(self, x):
        m = nn.ReLU()
        h11 = m(self.fc11(x))
        h11 = m(self.fc111(h11))
        h12 = m(self.fc12(x))
        h12 = m(self.fc222(h12))
        return self.fc21(h11), self.fc22(h12)
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    def decode(self, z):
        m = nn.Sigmoid()
        h31= m(self.fc3(z))
        h31= m(self.fc4(h31))
        h31_out = torch.bmm(h31.unsqueeze(2), h31.unsqueeze(1))
        h32 = m(self.fc32(z))
        h32 = m(self.fc5(h32))
        h32_out = torch.bmm(h32.unsqueeze(2), h32.unsqueeze(1))
        h33 = m(self.fc33(z))
        h33 = m(self.fc6(h33))
        h33_out = torch.bmm(h33.unsqueeze(2), h33.unsqueeze(1))
        h34 = m(self.fc34(z))
        h34 = m(self.fc7(h34))
        h34_out = torch.bmm(h34.unsqueeze(2), h34.unsqueeze(1))
        h35 = m(self.fc35(z))
        h35 = m(self.fc8(h35))
        h35_out = torch.bmm(h35.unsqueeze(2), h35.unsqueeze(1))
        m1 = nn.ReLU()
        h30 = m(h31_out + h32_out + h33_out + h34_out + h35_out)
        h30 = h30.view(-1, 68*68)
        h30 = self.fcintercept(h30)
        return h30.view(-1, 68*68), h31+h32+h33+h34+h35

    def fc_predict(self, mu):
        m = nn.Tanh()
        h1 = m(self.fc_pred1(mu))
        h1 = m(self.fc_pred2(h1))
        h1 = m(self.fc_pred3(h1))
        h1 = m(self.fc_pred4(h1))
        triu = (self.fc_pred5(h1))
        if self.training:
            return triu
        else: 
            triu_index = torch.triu(torch.ones(68,68), diagonal=1)
            fc = torch.zeros(mu.shape[0], 68*68)
            fc[:,torch.where(triu_index.flatten()==1)[0]] = triu
            fc = fc.reshape(mu.shape[0],68,68)
            fc = fc+torch.transpose(fc, dim0=1, dim1=2)
            return fc.reshape(mu.shape[0], 68*68)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon, x_latent = self.decode(z)
        fc_pred = self.fc_predict(z)
        return recon, mu, logvar, fc_pred

    def set_mask(self, masks):
        self.fc4.set_mask(masks[0])
        self.fc5.set_mask(masks[1])
        self.fc6.set_mask(masks[2])
        self.fc7.set_mask(masks[3])
        self.fc8.set_mask(masks[4])
        self.fcintercept.set_mask(masks[5])