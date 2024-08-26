"""
Train the PPO agent
"""
import argparse
import os
import pickle
from datetime import datetime
import numpy as np
from algo.ppo.PPO import PPO
from utils import make_env, set_random_seed
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch PPO Args')
parser.add_argument('--env_name', type=str, default='figure8',
                    help='name of the environment to run: ring, figure8, merge')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed')
parser.add_argument('--log_dir', type=str, default='./logs/')
parser.add_argument('--has_continuous_action_space', action='store_false',
                    help='continuous action space; else discrete')
parser.add_argument('--max_ep_len', type=int, default=1500, metavar='N',
                    help='max timesteps in one episode')
parser.add_argument('--max_training_steps', type=int, default=4e6, metavar='N',
                    help='break training loop if timeteps > max_training_timesteps')
parser.add_argument('--print_freq', type=int, default=6000, metavar='N',
                    help='print avg reward in the interval (in num timesteps)')
parser.add_argument('--log_freq', type=int, default=6000, metavar='N',
                    help='log avg reward in the interval (in num timesteps)')
parser.add_argument('--save_model_freq', type=int, default=6000, metavar='N',
                    help='save model frequency (in num timesteps)')
parser.add_argument('--update_timestep', type=int, default=6000, metavar='N',
                    help='update policy every n timesteps')
parser.add_argument('--action_std', type=float, default=0.6, metavar='G',
                    help='starting std for action distribution (Multivariate Normal)')
parser.add_argument('--action_std_decay_rate', type=float, default=0.05, metavar='G',
                    help='linearly decay action_std (action_std = action_std - action_std_decay_rate)')
parser.add_argument('--min_action_std', type=float, default=0.1, metavar='G',
                    help='minimum action_std (stop decay after action_std <= min_action_std)')
parser.add_argument('--action_std_decay_freq', type=int, default=int(2.5e5), metavar='N',
                    help='action_std decay frequency (in num timesteps)')
parser.add_argument('--K_epochs', type=int, default=80, metavar='N',
                    help='update policy for K epochs in one PPO update')
parser.add_argument('--eps_clip', type=float, default=0.2, metavar='G', help='clip parameter for PPO')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G', help='discount factor')
parser.add_argument('--lr_actor', type=float, default=0.0003, metavar='G',
                    help='learning rate for actor network')
parser.add_argument('--lr_critic', type=float, default=0.001, metavar='G',
                    help='learning rate for critic network')
args = parser.parse_args()


def train():
    print("==" * 40)
    print("training environment name : " + args.env_name)
    env = make_env()
    tag = 'rs{}'.format(args.seed)

    # state space dimension
    state_dim = env.observation_space.shape[0]

    # action space dimension
    if args.has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    writer = SummaryWriter(
        args.log_dir + '{}/PPO_{}_{}'.format(args.env_name, tag, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

    directory = "checkpoint"
    if not os.path.exists(directory):
        os.makedirs(directory)

    directory = directory + '/' + args.env_name + '/PPO_{}_{}/'.format(str.upper(tag),
                                                                       datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(directory + 'args.pkl', 'wb') as fp:
        pickle.dump(args, fp)

    checkpoint_path = directory + "/PPO_{}_{}.pth".format(args.env_name, args.seed)
    print("save checkpoint path : " + checkpoint_path)

    ############# print all hyperparameters #############
    print("--------------------------------------------------------------------------------------------")
    print("max training steps : ", args.max_training_steps)
    print("max timesteps per episode : ", args.max_ep_len)
    print("model saving frequency : " + str(args.save_model_freq) + " timesteps")
    print("log frequency : " + str(args.log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(args.print_freq) + " timesteps")
    print("--------------------------------------------------------------------------------------------")
    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)
    print("--------------------------------------------------------------------------------------------")
    if args.has_continuous_action_space:
        print("Initializing a continuous action space policy")
        print("--------------------------------------------------------------------------------------------")
        print("starting std of action distribution : ", args.action_std)
        print("decay rate of std of action distribution : ", args.action_std_decay_rate)
        print("minimum std of action distribution : ", args.min_action_std)
        print("decay frequency of std of action distribution : " + str(args.action_std_decay_freq) + " timesteps")
    else:
        print("Initializing a discrete action space policy")
    print("--------------------------------------------------------------------------------------------")
    print("PPO update frequency : " + str(args.update_timestep) + " timesteps")
    print("PPO K epochs : ", args.K_epochs)
    print("PPO epsilon clip : ", args.eps_clip)
    print("discount factor (gamma) : ", args.gamma)
    print("--------------------------------------------------------------------------------------------")
    print("optimizer learning rate actor : ", args.lr_actor)
    print("optimizer learning rate critic : ", args.lr_critic)
    if args.seed:
        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", args.seed)
        set_random_seed(args.seed, env)

    print("==" * 40)
    # initialize a PPO agent
    ppo_agent = PPO(state_dim, action_dim, args.lr_actor, args.lr_critic, args.gamma, args.K_epochs, args.eps_clip,
                    args.has_continuous_action_space, args.action_std)

    # track total training time
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("==" * 40)
    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    speed_list = []

    time_step = 0
    i_episode = 0

    # training loop
    while time_step <= args.max_training_steps:

        state = env.reset()
        current_ep_reward = 0

        for t in range(1, args.max_ep_len + 1):

            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, done, _ = env.step(action)

            speed_list.append(state[0] * 15)

            # saving reward and is_terminals
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)

            time_step += 1
            current_ep_reward += reward

            # update PPO agent
            if time_step % args.update_timestep == 0:
                ppo_agent.update()

            # if continuous action space; then decay action std of ouput action distribution
            if args.has_continuous_action_space and time_step % args.action_std_decay_freq == 0:
                ppo_agent.decay_action_std(args.action_std_decay_rate, args.min_action_std)

            # log in logging file
            if time_step % args.log_freq == 0:
                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_running_reward = 0
                log_running_episodes = 0

                writer.add_scalar('Reward/reward_batch', log_avg_reward, time_step)
                writer.add_scalar('Speed/speed_mean_batch', np.mean(speed_list), time_step)
                writer.add_scalar('Speed/speed_std_batch', np.std(speed_list), time_step)
                speed_list = []

            # printing average reward
            if time_step % args.print_freq == 0:
                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)

                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step,
                                                                                        print_avg_reward))

                print_running_reward = 0
                print_running_episodes = 0

            # save model weights
            if time_step % args.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                checkpoint_path = directory + "/PPO_{}_{}.pth".format(args.seed, i_episode)
                print("saving model at : " + checkpoint_path)
                ppo_agent.save(checkpoint_path)
                print("model saved")
                print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                print("--------------------------------------------------------------------------------------------")

            # break; if the episode is over
            if done:
                break

        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1

    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")


if __name__ == '__main__':
    train()
