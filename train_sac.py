"""
Train the SAC agent
"""
import os
import argparse
from datetime import datetime
import pickle
import numpy as np
import itertools
from algo.sac.sac import SAC
from algo.sac.replay_memory import ReplayMemory
from utils import make_env, set_random_seed
from torch.utils.tensorboard import SummaryWriter


parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env_name', type=str, default='figure8',
                    help='name of the environment to run: ring, figure8, merge')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                    help='Automaically adjust α')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--max_training_steps', type=int, default=4e6, metavar='N',
                    help='maximum number of steps')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA')
args = parser.parse_args()

# Environment
print(args)
env = make_env()

set_random_seed(args.seed, env)

tag = 'rs{}'.format(args.seed)
print('Random seed: {}'.format(args.seed))
checkpoints_path = './checkpoint/{}/SAC_{}_{}/'.format(args.env_name, str.upper(tag),
                                                       datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)

with open(checkpoints_path+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

writer = SummaryWriter(
    'logs/{}/SAC_{}_{}_{}_{}'.format(args.env_name, tag, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
                                     args.policy, "autotune" if args.automatic_entropy_tuning else ""))

print('Checkpoints path: {}'.format(checkpoints_path))

# Memory
memory = ReplayMemory(args.replay_size)

# Training Loop
total_numsteps = 0
updates = 0
reward_list = []
speed_mean_list = []
speed_std_list = []

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    episode_speed = []
    done = False
    state = env.reset()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = agent.update_parameters(memory,
                                                                                                     args.batch_size,
                                                                                                     updates)
                updates += 1

        next_state, reward, done, _ = env.step(action)  # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        episode_speed.append(state[0] * 15)

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env.env_params.horizon else float(not done)

        memory.push(state, action, reward, next_state, mask)  # Append transition to memory

        state = next_state

    if total_numsteps > args.max_training_steps + 1:
        break

    reward_list.append(episode_reward)
    speed_mean_list.append(np.mean(episode_speed))
    speed_std_list.append(np.std(episode_speed))
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps,
                                                                                  episode_steps,
                                                                                  round(episode_reward, 2)))

    if i_episode % 4 == 0 and args.eval == True:
        writer.add_scalar('Reward/reward_batch', np.mean(reward_list), total_numsteps)
        writer.add_scalar('Speed/speed_mean_batch', np.mean(speed_mean_list), total_numsteps)
        writer.add_scalar('Speed/speed_std_batch', np.mean(speed_std_list), total_numsteps)

        reward_list = []
        speed_mean_list = []
        speed_std_list = []

        agent.save_model(checkpoints_path, suffix=total_numsteps)

env.close()
writer.close()
