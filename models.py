import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import seed_everything
class MixedApproachTiedAutoEncoder(nn.Module):
    def __init__(self, inp, out,SEED,bias=True):
        super().__init__()
        seed_everything(SEED)#seed_everythingをあることでmodelの初期値が固定されるはず
        self.encoder = nn.Linear(inp, out, bias=bias)
        #self.bias = nn.Parameter(torch.rand(59412)*0.001)
    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = F.linear(encoded_feats, self.encoder.weight.t())
        return encoded_feats, reconstructed_output
    
    
class MixedApproachTiedMultiAutoEncoder(nn.Module):
    def __init__(self, inp,xx,out,SEED,bias=True):
        super().__init__()
        self.encoder = nn.Linear(inp, xx, bias=bias)
        self.encoder2 = nn.Linear(xx, out, bias=bias)
        self.prelu = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.biasxx = nn.Parameter(torch.rand(xx)*0.001)
        self.bias = nn.Parameter(torch.rand(inp)*0.001)
    def forward(self, input):
        encoded_feats = self.prelu(self.encoder(input))
        encoded_feats = self.encoder2(encoded_feats)
        reconstructed_output = self.prelu2(F.linear(encoded_feats, self.encoder2.weight.t()) + self.biasxx)
        reconstructed_output = F.linear(reconstructed_output, self.encoder.weight.t()) + self.bias
        
        return encoded_feats, reconstructed_output
    
    
class AutoEncoder(nn.Module):
    def __init__(self, inp, out,SEED,bias=True):
        super().__init__()
        seed_everything(SEED)#seed_everythingをあることでmodelの初期値が固定されるはず
        self.encoder = nn.Linear(inp, out, bias=bias)
        self.decoder = nn.Linear(out, inp, bias=bias)
        
    def forward(self, input):
        encoded_feats = self.encoder(input)
        reconstructed_output = self.decoder(encoded_feats)
        return encoded_feats, reconstructed_output

    
    
__factory = {
    'Simple':MixedApproachTiedAutoEncoder,
    'Multi':MixedApproachTiedMultiAutoEncoder,
    'Asymmetric':AutoEncoder,
}
    
def create(modeltype,inp,out,SEED,xx=None):
    
    if modeltype not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(modeltype))
    model = __factory[modeltype]
    if xx is not None:
        return model(inp,xx,out,SEED)
    else:
        return model(inp,out,SEED)
    