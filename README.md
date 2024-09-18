# traffic-expertise-RL
This repo provides code for our paper: [Traffic expertise meets residual RL: Knowledge-informed model-based residual reinforcement learning for CAV trajectory control](https://arxiv.org/abs/2408.17380).

We are currently in the process of organizing our code and preparing it for release.

Stay tuned for our upcoming open-source project on GitHub!

## Introduction
This paper introduces a knowledge-informed model-based residual reinforcement learning framework aimed at enhancing learning efficiency by infusing established traffic domain knowledge into the learning process and avoiding the issue of beginning from zero. 

<div align=center><img src=./assets/poster.png ></div>


### Demonstration video
https://github.com/zihaosheng/traffic-expertise-RL/assets/48112700/67613b2c-4b3c-4b7a-b8b0-031ceb2632a1


## Devkit setup
#### 1. Create conda environment
```shell
# Clone the code to local
git clone https://github.com/zihaosheng/traffic-expertise-RL.git
cd traffic-expertise-RL

# Create virtual environment
conda create -n teRL python=3.8
conda activate teRL

# Install basic dependency
pip install -r requirements.txt

# Install torch
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

#### 2. Install SUMO
The SUMO version used in this paper is 1.8.0.

To finalize your setup, please make sure to set the `SUMO_HOME` environment variable and have it point to
the directory of your SUMO installation.
```shell
which sumo
sumo --version
sumo-gui
```
More details can be found: https://sumo.dlr.de/docs/Installing/index.html


#### 3. Download Flow
```bash
git clone https://github.com/flow-project/flow.git
cd flow
```
Revise the code from line 508 in `flow/core/kernel/network/traci.py`:

<details>
  <summary>Click to expand/collapse the code</summary>

```python
subprocess.call(
    'netconvert -c ' + self.net_path + self.cfgfn +
    ' --output-file=' + self.cfg_path + self.netfn +
    ' --no-internal-links="false"',
    stdout=subprocess.DEVNULL,
    shell=True)
```
</details>

Revise the code in `flow/controllers/rlcontroller.py`:
<details>
  <summary>Click to expand/collapse the code</summary>
  
```python
"""Contains the RLController class."""
import numpy as np
from flow.controllers.base_controller import BaseController


class RLController(BaseController):
    """RL Controller.

    Vehicles with this class specified will be stored in the list of the RL IDs
    in the Vehicles class.

    Usage: See base class for usage example.

    Attributes
    ----------
    veh_id : str
        Vehicle ID for SUMO identification

    Examples
    --------
    A set of vehicles can be instantiated as RL vehicles as follows:

        >>> from flow.core.params import VehicleParams
        >>> vehicles = VehicleParams()
        >>> vehicles.add(acceleration_controller=(RLController, {}))

    In order to collect the list of all RL vehicles in the next, run:

        >>> from flow.envs import Env
        >>> env = Env(...)
        >>> rl_ids = env.k.vehicle.get_rl_ids()
    """


    def __init__(self, veh_id, car_following_params):
        """Instantiate PISaturation."""
        BaseController.__init__(self, veh_id, car_following_params, delay=1.0)

        # maximum achievable acceleration by the vehicle
        self.max_accel = car_following_params.controller_params['accel']

        # history used to determine AV desired velocity
        self.v_history = []

        # other parameters
        self.gamma = 2
        self.g_l = 7
        self.g_u = 30
        self.v_catch = 1

        # values that are updated by using their old information
        self.alpha = 0
        self.beta = 1 - 0.5 * self.alpha
        self.U = 0
        self.v_target = 0
        self.v_cmd = 0

    def get_accel(self, env):
        """See parent class."""
        lead_id = env.k.vehicle.get_leader(self.veh_id)
        lead_vel = env.k.vehicle.get_speed(lead_id)
        this_vel = env.k.vehicle.get_speed(self.veh_id)

        dx = env.k.vehicle.get_headway(self.veh_id)
        dv = lead_vel - this_vel
        dx_s = max(2 * dv, 4)

        # update the AV's velocity history
        self.v_history.append(this_vel)

        if len(self.v_history) == int(38 / env.sim_step):
            del self.v_history[0]

        # update desired velocity values
        v_des = np.mean(self.v_history)
        v_target = v_des + self.v_catch \
            * min(max((dx - self.g_l) / (self.g_u - self.g_l), 0), 1)

        # update the alpha and beta values
        alpha = min(max((dx - dx_s) / self.gamma, 0), 1)
        beta = 1 - 0.5 * alpha

        # compute desired velocity
        self.v_cmd = beta * (alpha * v_target + (1 - alpha) * lead_vel) \
            + (1 - beta) * self.v_cmd

        # compute the acceleration
        accel = (self.v_cmd - this_vel) / env.sim_step

        return min(accel, self.max_accel)
```
</details>

## Main experiments
### Train your model
```shell
python train_terl.py --max_training_steps=4000000 --seed=1234
```
### Train RL baselines
```shell
python train_[ppo/sac/trpo].py --max_training_steps=4000000 --seed=1234
```
The results of these scripts can be visualized using TensorBoard.
```shell
tensorboard --logdir=logs --port=6006
```
### Train virtual environment model
#### Download data
To demonstrate the effectiveness and superiority of the Knowledge NN, we utilize the dataset developed by 
[Mo et al. (2021)](https://www.sciencedirect.com/science/article/pii/S0968090X21002539), 
which encompasses four representative driving scenarios: `acceleration, deceleration, cruising, and emergency braking`. 
In total, the dataset consists of 262,630 data points, with 70% randomly selected for training, 10% for validation, 
and the remaining 20% for testing.
Raw data can be downloaded [here](https://github.com/CU-DitecT/PINN-CFM/blob/main/), 
or our processed data [here](./data).

#### Train models
Train the Knowledge NN and the Vanilla NN and compare the loss during training and validation.
```shell
python train_compare_loss.py --seed 1234 --idm_data_path ./data/idm_data.h5 
```
Compare the prediction performance between the Knowledge NN and the Vanilla NN under different dataset sizes.
```shell
python train_compare_datesize.py --seed 1234 --idm_data_path ./data/idm_data.h5 
```
The results of these scripts can also be viewed using TensorBoard.

## Other Awesome Projects from Our Team

Our team is actively involved in various innovative projects in the realm of autonomous driving. Here are some other exciting repositories that you might find interesting:

- **[Physics-enhanced RLHF](https://github.com/zilin-huang/PE-RLHF)**
- **[Human as AI mentor](https://zilin-huang.github.io/HAIM-DRL-website/)**
  

## Reference
```latex
@article{
    sheng2024traffic,
    title={Traffic expertise meets residual RL: Knowledge-informed model-based residual reinforcement learning for CAV trajectory control},
    author={Sheng, Zihao and Huang, Zilin and Chen, Sikai},
    journal={Communications in Transportation Research},
    year={2024},
}
