#@title 6. Define Model Architectures
import torch
import torch.nn as nn

def get_activation(name): return nn.PReLU() if name == 'prelu' else nn.ReLU() if name == 'relu' else nn.GELU()

class Block1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2, kernel_size=3, activation='prelu'):
        super().__init__(); self.b = nn.Sequential(nn.Conv1d(in_channels,out_channels,kernel_size,stride,padding=kernel_size//2), nn.BatchNorm1d(out_channels), get_activation(activation), nn.Conv1d(out_channels,out_channels,kernel_size,1,padding=kernel_size//2), nn.BatchNorm1d(out_channels), get_activation(activation))
    def forward(self, x): return self.b(x)

class UNetEncoder1D(nn.Module):
    def __init__(self, in_channels, activation='prelu'):
        super().__init__(); self.start=Block1D(in_channels,32,stride=1,activation=activation); self.e1=Block1D(32,64,stride=2,activation=activation); self.e2=Block1D(64,128,stride=2,activation=activation); self.e3=Block1D(128,256,stride=2,activation=activation); self.mid=Block1D(256,512,stride=2,activation=activation)
    def forward(self, x): x=x.squeeze(1).permute(0,2,1); s1=self.start(x); s2=self.e1(s1); s3=self.e2(s2); s4=self.e3(s3); return self.mid(s4)

class Project(nn.Module):
    def __init__(self,i,o): super().__init__(); self.l=nn.Linear(i,o)
    def forward(self,x): return self.l(x.flatten(1))

class Query(nn.Module):
    def __init__(self,s,d): super().__init__(); self.q=nn.Parameter(torch.randn(1,s,d))
    def forward(self,x): return self.q.repeat(x.shape[0],1,1)

class Transformer(nn.Module):
    def __init__(self,i,n,d): super().__init__(); self.t=nn.TransformerEncoderLayer(d_model=i,nhead=n,dropout=d,batch_first=True,dim_feedforward=i*4)
    def forward(self,q,c): return self.t(q)

class W2WTransformerModel(nn.Module):
    def __init__(self,c):
        super().__init__()
        p=c['finetuning']['model_params']
        # Dynamically calculate the flattened feature size
        with torch.no_grad():
            dummy_encoder = UNetEncoder1D(p['in_channels'], p['act_name'])
            dummy_output = dummy_encoder(torch.randn(1, 1, p['patch_height'], p['in_channels']))
            p_in = dummy_output.flatten(1).shape[1]

        self.encoder = UNetEncoder1D(p['in_channels'], p['act_name'])
        self.project = Project(p_in, p['hidden_dim'])
        self.query = Query(p['num_queries'], p['hidden_dim'])
        self.transformers = nn.ModuleList([Transformer(p['hidden_dim'], p['num_heads'], p['dropout']) for _ in range(p['num_transformers'])])
        self.finalize = nn.Sequential(nn.Linear(p['hidden_dim'], p['output_size']), get_activation(p['act_name']), nn.LayerNorm(p['output_size']))

    def forward(self,img):
        seq = self.project(self.encoder(img)).unsqueeze(1)
        q = self.query(seq)
        for t in self.transformers: q = t(q, seq)
        return self.finalize(q)

print("✅ Model architectures defined.")