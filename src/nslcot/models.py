import torch
from torch import nn

class FeatureClassifier(nn.Module):
    def __init__(self,input_dim,hidden_dim=256,kind='gru',steps=3):
        super().__init__();self.kind=kind;self.steps=steps
        self.projection=nn.Linear(input_dim,hidden_dim)
        if kind=='gru':self.refine=nn.GRUCell(hidden_dim,hidden_dim)
        elif kind=='mlp':self.refine=nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.GELU())
        elif kind=='matched_mlp':
            # Width 768 gives ~394k refinement parameters vs GRU's ~395k.
            self.refine=nn.Sequential(nn.Linear(hidden_dim,hidden_dim*3),nn.GELU(),nn.Linear(hidden_dim*3,hidden_dim),nn.GELU())
        elif kind=='linear':self.refine=nn.Identity()
        else:raise ValueError(kind)
        self.head=nn.Linear(hidden_dim,2)
    def forward(self,h):
        x=self.projection(h);z=x
        if self.kind=='gru':
            for _ in range(self.steps):z=self.refine(x,z)
        else:z=self.refine(z)
        return self.head(z)
    @property
    def trainable_parameters(self):return sum(p.numel() for p in self.parameters() if p.requires_grad)
