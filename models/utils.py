from torch import nn

def get_activation(name):
    if name == 'relu':
        act = nn.ReLU()
    elif name == 'tanh':
        act = nn.Tanh()
    elif name == 'sigmoid':
        act = nn.Sigmoid()
    elif name == 'gelu':
        act = nn.GELU()
    elif name == 'prelu':
        act = nn.PReLU()
    elif name == 'elu':
        act = nn.ELU()
    elif name == 'lrelu':
        act = nn.LeakyReLU()
    return act

        