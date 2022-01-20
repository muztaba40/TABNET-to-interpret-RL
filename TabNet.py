import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.Sparsemax import *
import numpy as np

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparaeters:
'''n_d, n_a: 8 to 512
batch size: 256 to 32768
virtual batch size: 128 to 2048
sparsity regularization constant: 0 to 0.00001
number of shared GLU Blocks: 2 to 10
number of independent decision Blocks: 2 to 10
relaxation constant: 1 to 2.5
number of decision steps: 2 to 10
batch normalization momentum: 0.5 to 0.98
'''
# Ghost Batch Normalization:

class GBN(nn.Module):
    '''to train large batches of data and generalize better at same time. Split input batch into equal-sized sub-batches(virtual batch size)
    and apply same Batch Normalization layer on them. GBN applied to input features.'''

    '''def __init__(self, input_dim, vbs=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.vbs = vbs
        self.bn = nn.BatchNorm2d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.vbs)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)'''
    def __init__(self,inp,vbs=128,momentum=0.01):
        super().__init__()
        self.bn = nn.BatchNorm1d(inp,momentum=momentum)
        self.vbs = vbs
    def forward(self,x):
        chunk = torch.chunk(x,x.size(0)//self.vbs,0)
        res = [self.bn(y) for y in chunk]
        return torch.cat(res,0)

    
# Sparsemax:
'''Compared to softmax. sparsemax to project the mask for the feature selection step onto a sparser space. Implementation: 
https://github.com/dreamquark-ai/tabnet/blob/develop/pytorch_tabnet/sparsemax.py'''

# Attention Transformer:

class AttentionTransformer(nn.Module):
    '''The models learns the relationship between relevant features and 
    decides which features to pass on to the feature transformer of the current decision step. 
    Each Attention Transformer consists of a fully connected layer, 
    a Ghost Batch Normalization Layer, and a Sparsemax layer. 
    The attention transformer in each decision step receives the input features,
    processed features from the previous step and prior information about used-features.
    The prior information is represented by a matrix of size batch_size x input_features.
    Also a relaxation parameter that limits how many times a certain feature can be used in a forward pass.'''

    def __init__(self,d_a,inp_dim,out_dim,relax,vbs=128):
        super().__init__()
        self.fc = nn.Linear(d_a,inp_dim)
        self.bn = GBN(out_dim,vbs=vbs)
        self.smax = Sparsemax()
        self.r = relax
    #a:feature from previous decision step
    def forward(self,a,priors): 
        a = self.bn(self.fc(a)) 
        mask = self.smax(a*priors) 
        priors =priors*(self.r-mask)  #updating the prior
        return mask
    
# Feature Transformer:
    
    ###### Gated Linear Unit:
class GLU(nn.Module):
    '''First double the dimension of the input features to the GLU using a fully connected layer.
    Normalize the resultant matrix using a GBN Layer.'''
    def __init__(self,inp_dim,out_dim,fc=None,vbs=128):
        super().__init__()
        if fc:
            self.fc = fc
        else:
            self.fc = nn.Linear(inp_dim,out_dim*2)
        self.bn = GBN(out_dim*2,vbs=vbs) 
        self.od = out_dim
    def forward(self,x):
        x = self.bn(self.fc(x))
        return x[:,:self.od]*torch.sigmoid(x[:,self.od:])
    
class FeatureTransformer(nn.Module):
    '''all the selected features are processed to generate the final output. 
    Each feature transformer is composed of multiple Gated Linear Unit Blocks. apply a sigmoid to the second half of the resultant features 
    and multiply the results to the first half. The result is multiplied with a scaling factor(sqrt(0.5) in this case) and added to the input. 
    This summed result is the input for the next GLU Block in the sequence.'''
    def __init__(self,inp_dim,out_dim,shared,n_ind,vbs=128):
        super().__init__()
        first = True
        self.shared = nn.ModuleList()
        if shared:
            self.shared.append(GLU(inp_dim,out_dim,shared[0],vbs=vbs))
            first= False    
            for fc in shared[1:]:
                self.shared.append(GLU(out_dim,out_dim,fc,vbs=vbs))
        else:
            self.shared = None
        self.independ = nn.ModuleList()
        if first:
            self.independ.append(GLU(inp_dim,out_dim,vbs=vbs))
        for x in range(first, n_ind):
            self.independ.append(GLU(out_dim,out_dim,vbs=vbs))
        self.scale = torch.sqrt(torch.tensor([.5],device=device))
    def forward(self,x):
        if self.shared:
            x = self.shared[0](x)
            for glu in self.shared[1:]:
                x = torch.add(x, glu(x))
                x = x*self.scale
        for glu in self.independ:
            x = torch.add(x, glu(x))
            x = x*self.scale
        return x
    
# Decision Step: 
class DecisionStep(nn.Module):
    '''Combine the Attention Transformer and Feature Transformer'''
    def __init__(self,inp_dim,n_d,n_a,shared,n_ind,relax,vbs=128):
        super().__init__()
        self.fea_tran = FeatureTransformer(inp_dim,n_d+n_a,shared,n_ind,vbs)
        self.atten_tran =  AttentionTransformer(n_a,inp_dim,relax,vbs)
    def forward(self,x,a,priors):
        mask = self.atten_tran(a,priors)
        sparse_loss = ((-1)*mask*torch.log(mask+1e-10)).mean()
        x = self.fea_tran(x*mask)
        return x,sparse_loss    
    
# TabNet:
class TabNet(nn.Module):
    '''These features are processed together until they reach the splitter. The ReLU activation is applied on the n_d dimensioned vector. 
    The outputs of all the decision steps are summed together and passed through a fully connected layer to map them to the output dimension.'''
    def __init__(self,inp_dim,final_out_dim,n_d=64,n_a=64,
                n_shared=2,n_ind=2,n_steps=5,relax=1,vbs=128):
        super().__init__()
        if n_shared>0:
            self.shared = nn.ModuleList()
            self.shared.append(nn.Linear(inp_dim,2*(n_d+n_a)))
            for x in range(n_shared-1):
                self.shared.append(nn.Linear(n_d+n_a,2*(n_d+n_a)))
        else:
            self.shared=None
        self.first_step = FeatureTransformer(inp_dim,n_d+n_a,self.shared,n_ind) 
        self.steps = nn.ModuleList()
        for x in range(n_steps-1):
            self.steps.append(DecisionStep(inp_dim,n_d,n_a,self.shared,n_ind,relax,vbs))
        self.fc = nn.Linear(n_d,final_out_dim)
        self.bn = nn.BatchNorm1d(inp_dim)
        self.n_d = n_d
    def forward(self,x):
        x = self.bn(x)
        x_a = self.first_step(x)[:,self.n_d:]
        sparse_loss = torch.zeros(1).to(x.device)
        out = torch.zeros(x.size(0),self.n_d).to(x.device)
        priors = torch.ones(x.shape).to(x.device)
        for step in self.steps:
            x_te,l = step(x,x_a,priors)
            out += F.relu(x_te[:,:self.n_d])
            x_a = x_te[:,self.n_d:]
            sparse_loss += l
        return self.fc(out),sparse_loss