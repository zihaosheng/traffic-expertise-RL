import torch
import torch.nn as nn
import numpy as np


class NN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=(128,128), activation='tanh', learning_rate=0.0001):
        super(NN, self).__init__()
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
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
        x = self.action(x)
        return x


class FakeEnv:
    def __init__(self, sim_step, max_speed, max_length, input_dim=3, output_dim=1, hidden_size=(64, 128, 32), activation='tanh',
                 params_value=None, params_trainable=None, device=None):
        self.sim_step = sim_step
        self.max_speed = max_speed
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.activation = activation
        self.params_value = params_value
        self.params_trainable = params_trainable
        self.device = device
        self.state_nn = NN(input_dim+output_dim, input_dim)
        self.reward_nn = NN(input_dim+output_dim, 1)

    def step(self, obs, act):
        obs = torch.tensor([obs], dtype=torch.double, device=self.device)
        act = torch.tensor([act], dtype=torch.double, device=self.device)
        reward = self.reward_nn(torch.cat((obs, act), dim=1))
        next_obs = self.state_nn(torch.cat((obs, act), dim=1))

        return next_obs.data[0].numpy(), reward.data[0].numpy()[0]

    def train_model(self, batch, batch_size=64, max_iter=20):
        global loss_state, loss_reward
        rewards = torch.tensor(batch.reward, dtype=torch.double, device=self.device).view(-1, 1)  # Tensor (200, )
        actions = torch.tensor(np.concatenate(batch.action, 0), dtype=torch.double, device=self.device).view(-1,1)  # Tensor (200, )
        states = torch.tensor(batch.state, dtype=torch.double, device=self.device)  # Tensor (200, 3)
        next_states = torch.tensor(batch.next_state, dtype=torch.double, device=self.device)

        permutation = np.random.permutation(states.shape[0])
        rewards, actions, states, next_states = rewards[permutation], actions[permutation], states[permutation], next_states[permutation]

        for epoch in range(max_iter):
            train_index = np.random.permutation(states.shape[0])

            for batch_start_pos in range(0, states.shape[0], batch_size):
                batch_index = train_index[batch_start_pos:batch_start_pos + batch_size]
                batch_states = states[batch_index]
                batch_actions = actions[batch_index]
                batch_next_states = next_states[batch_index]
                batch_rewards = rewards[batch_index]

                next_obs = self.state_nn(torch.cat((batch_states, batch_actions), dim=1))
                loss_state = torch.mean(torch.pow(batch_next_states - next_obs, 2), dim=0)
                loss_state = torch.sum(loss_state)
                self.state_nn.optimizer.zero_grad()
                loss_state.backward()
                self.state_nn.optimizer.step()

                rewards_prediction = self.reward_nn(torch.cat((batch_states, batch_actions), dim=1))
                loss_reward = torch.mean(torch.pow(batch_rewards - rewards_prediction, 2), dim=0)
                self.reward_nn.optimizer.zero_grad()
                loss_reward.backward()
                self.reward_nn.optimizer.step()
            print('Epoch %d, loss_state: %f, loss_reward: %f' % (epoch, loss_state.item(), loss_reward.item()))

        return loss_state.item(), loss_reward.item()


if __name__ == '__main__':

    params_value = dict(v0=30, T=1, a=1, b=1.5, delta=4, s0=2, )
    print(params_value)

