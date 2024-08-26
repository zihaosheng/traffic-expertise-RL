import numpy as np
import random
import torch

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams, SumoCarFollowingParams
from flow.controllers import IDMController, ContinuousRouter, RLController
from flow.networks.figure_eight import FigureEightNetwork, ADDITIONAL_NET_PARAMS
from algo.figure_eight_env import Figure8POEnv, ADDITIONAL_ENV_PARAMS


def set_random_seed(seed=1234, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)


class Feeder(torch.utils.data.Dataset):
    def __init__(self, data, train_val_test='train'):
        features = data[:, :3]
        features = features.astype(np.float32)
        labels = data[:, [3]]
        labels = labels.astype(np.float32)
        total_num = features.shape[0]

        permutation = np.random.permutation(total_num)
        features, labels = features[permutation], labels[permutation]

        train_idx_list = list(np.arange(0, int(total_num * 0.7)))
        val_idx_list = list(np.arange(int(total_num * 0.7), int(total_num * 0.8)))
        test_idx_list = list(np.arange(int(total_num * 0.8), total_num))
        if train_val_test.lower() == 'train':
            self.features = features[train_idx_list]
            self.labels = labels[train_idx_list]
        elif train_val_test.lower() == 'val':
            self.features = features[val_idx_list]
            self.labels = labels[val_idx_list]
        elif train_val_test.lower() == 'test':
            self.features = features[test_idx_list]
            self.labels = labels[test_idx_list]
        else:
            raise ValueError("Invalid parameter configuration: 'train_val_test' has an incorrect "
                             "value. Please ensure that this parameter is correctly set.")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def make_env(horizon=1500, warmup_steps=0, render=False):
    ADDITIONAL_ENV_PARAMS['radius_ring'] = [32, 35]
    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS)
    initial_config = InitialConfig(spacing="uniform", bunching=50)
    sim_params = SumoParams(sim_step=0.1, render=render, emission_path=None, color_by_speed=False)
    env_params = EnvParams(horizon=horizon, warmup_steps=warmup_steps, additional_params=ADDITIONAL_ENV_PARAMS)

    print(ADDITIONAL_NET_PARAMS)
    print(ADDITIONAL_ENV_PARAMS)

    name = "figure_eight"
    vehicles = VehicleParams()
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {
            "noise": 0.2
        }),
        car_following_params=SumoCarFollowingParams(
            min_gap=0
        ),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=13,
        color='white'
    )
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=1,
        color='red'
    )

    network = FigureEightNetwork(name, vehicles, net_params, initial_config)

    env = Figure8POEnv(env_params, sim_params, network)
    return env


if __name__ == "__main__":
    env = make_env(render=True)

    state = env.reset()
    for i in range(3000):
        a = env.action_space.sample()
        # print(a)
        next_state, reward, done, _ = env.step(None)
        print(next_state)
