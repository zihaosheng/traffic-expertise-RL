import torch
import torch.nn as nn


class IDM(nn.Module):
    def __init__(self, params_value=None, params_trainable=None, device=torch.device("cpu")):
        super(IDM, self).__init__()
        if params_value is None:
            params_value = dict(v0=30, T=1, a=1, b=1.5, delta=4, s0=2, )
        if params_trainable is None:
            params_trainable = dict(v0=False, T=False, a=False, b=False, delta=False, s0=False, )
        self.torch_params = dict()
        for k, v in params_value.items():
            if params_trainable[k] is True:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device),
                                                               requires_grad=True,
                                                               )
                self.torch_params[k].retain_grad()
            else:
                self.torch_params[k] = torch.nn.Parameter(torch.tensor(v, dtype=torch.float32, device=device),
                                                          requires_grad=False)
        self.veh_length = 0.

    def forward(self, x):
        dx = x[:, 0]
        dv = x[:, 1]
        v = x[:, 2]

        s0 = self.torch_params['s0']
        v0 = self.torch_params['v0']
        T = self.torch_params['T']
        a = self.torch_params['a']
        b = self.torch_params['b']
        delta = self.torch_params['delta']

        s_star = s0 + torch.clamp_min(T*v - v * dv / (2 * torch.sqrt(a*b)), 0)
        acc = a * (1 - torch.pow(v/v0, delta) - torch.pow(s_star/(dx-self.veh_length), 2))
        return acc.view(-1, 1)


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(128,128), activation='tanh', learning_rate=0.0001, device=torch.device('cpu')):
        super(NN, self).__init__()
        self.device = device
        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()

        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_size:
            self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

        self.action = nn.Linear(last_dim, output_dim)
        self.to(self.device)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = x.to(self.device)
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.action(x)
        return x


class KnowledgeNN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(128, 128), activation='relu', params_value=None,
                 params_trainable=None, device=torch.device('cpu'), learning_rate=0.0001):
        super(KnowledgeNN, self).__init__()

        self.physics_model = IDM(params_value, params_trainable, device)
        self.nn_model = NN(input_dim, output_dim, hidden_size, activation, device=device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        phy = self.physics_model(obs)
        res = self.nn_model(obs)
        return phy + res

    def phy(self, dx, dv, v_fv):
        phy = self.physics_model(dx, dv, v_fv)
        return phy


if __name__ == '__main__':

    device = torch.device("cuda:0")
    pinn = KnowledgeNN(3, 1, device=device)

    input = torch.tensor([[6, 0, 5],[1, 0, 5]], dtype=torch.float32, device=device)
    acc = pinn(input)
    print(acc)

    # IDM
    params_value = dict(v0=30, T=1, a=1, b=1.5, delta=4, s0=2, )
    params_value2 = {
        "v0": 30, "T": 1, "a": 1, "b": 1.5, "delta": 4, "s0": 2
    }
    print(params_value)
    print(params_value2)

    idm = IDM(device=torch.device("cuda:0"))

    input = torch.tensor([[6, 0, 5],[6, 0, 5]], device=torch.device("cuda:0"))
    acc = idm(input)
    print(acc)