import torch
from torch import nn
from config import DROPOUT
class NIDSDNN(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(input_dim,128),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(128,64),nn.ReLU(),nn.Dropout(DROPOUT),nn.Linear(64,32),nn.ReLU(),nn.Linear(32,1))
    def forward(self,x): return self.net(x).squeeze(1)

def load_model(path,input_dim,device='cpu'):
    model=NIDSDNN(input_dim).to(device)
    state=torch.load(path,map_location=device,weights_only=True)
    model.load_state_dict(state); model.eval(); return model
