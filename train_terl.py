"""
Train the teRL agent
"""
import argparse
import os
import pickle
from datetime import datetime
import scipy.optimize
from torch.utils.tensorboard import SummaryWriter

from algo.fake_env import FakeEnv
from algo.trpo.models import *
from algo.trpo.replay_memory import Memory
from algo.trpo.running_state import ZFilter
from algo.trpo.trpo_utils import *
from utils import make_env, set_random_seed

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True

torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch teRL Args')
parser.add_argument('--env_name', type=str, default='figure8',
                    help='name of the environment to run: ring, figure8, merge')
parser.add_argument('--seed', type=int, default=1234, metavar='N',
                    help='random seed')
parser.add_argument('--gamma', type=float, default=0.995, metavar='G',
                    help='discount factor')
parser.add_argument('--tau', type=float, default=0.97, metavar='G',
                    help='gae')
parser.add_argument('--l2-reg', type=float, default=1e-3, metavar='G',
                    help='l2 regularization regression')
parser.add_argument('--max-kl', type=float, default=1e-2, metavar='G',
                    help='max kl value')
parser.add_argument('--damping', type=float, default=1e-1, metavar='G',
                    help='damping')
parser.add_argument('--reward_threshold', type=float, default=0, metavar='G',
                    help='reward threshold for the virtual environment model')
parser.add_argument('--k_max', type=int, default=500, metavar='N',
                    help='upper bound for rollout step')
parser.add_argument('--rollout_sensitivity', type=float, default=2, metavar='G',
                    help='rollout length sensitivity')
parser.add_argument('--loss_state_threshold', type=float, default=1e-2, metavar='G')
parser.add_argument('--reward_state_threshold', type=float, default=1, metavar='G')
parser.add_argument('--batch-size', type=int, default=6000, metavar='N',
                    help='batch size')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs')
parser.add_argument('--max_training_steps', type=int, default=4e6, metavar='N',
                    help='maximum training steps')
args = parser.parse_args()
print(args)
env = make_env()
set_random_seed(args.seed, env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

# log with tensorboard
tag = 'rs{}'.format(args.seed)
print('Random seed: {}'.format(args.seed))
writer = SummaryWriter(
    'logs/' + '{}/TERL_{}_{}'.format(args.env_name, tag, datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))

checkpoints_path = './checkpoint/{}/TERL_{}_{}/'.format(args.env_name, str.upper(tag),
                                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
if not os.path.exists(checkpoints_path):
    os.makedirs(checkpoints_path)
print('Log path: {}'.format(writer.log_dir))
print('Checkpoints path: {}'.format(checkpoints_path))

with open(checkpoints_path + 'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

fake_env = FakeEnv(env.sim_step, 15., env.env_params.additional_params['radius_ring'][1],
                   device=torch.device('cpu'))


def select_action(state):
    state = torch.from_numpy(state).unsqueeze(0)
    action_mean, _, action_std = policy_net(Variable(state))
    action = torch.normal(action_mean, action_std)
    return action


def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))
    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0), 1)
    deltas = torch.Tensor(actions.size(0), 1)
    advantages = torch.Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]

        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    # Original code uses the same LBFGS to optimize the value loss
    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

        values_ = value_net(Variable(states))

        value_loss = (values_ - targets).pow(2).mean()

        # weight decay
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, opt_info = scipy.optimize.fmin_l_bfgs_b(get_value_loss,
                                                            get_flat_params_from(value_net).double().numpy(),
                                                            maxiter=25)
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / advantages.std()

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        if volatile:
            with torch.no_grad():
                action_means, action_log_stds, action_stds = policy_net(Variable(states))
        else:
            action_means, action_log_stds, action_stds = policy_net(Variable(states))

        log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))

        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)


running_state = ZFilter((num_inputs,), clip=5)
running_reward = ZFilter((1,), demean=False, clip=10)

total_step = 0
i_episode = 0
while total_step < args.max_training_steps:
    memory = Memory()

    num_steps = 0
    reward_batch = []
    num_episodes = 0
    speed_mean_batch = []
    speed_std_batch = []
    print(">>>>>>>>>>>>>>> Interact With Real Env <<<<<<<<<<<<<<<<<<<<")
    while num_steps < args.batch_size:
        state = env.reset()
        state = running_state(state)

        rl_id = env.k.vehicle.get_rl_ids()[0]
        reward_sum = 0
        speed_list = []
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            rl_action = action.data[0].numpy()

            # get action from established traffic domain knowledge
            phy_action = env.k.vehicle.get_acc_controller(rl_id).get_accel(env)

            action = rl_action + phy_action

            next_state, reward, done, _ = env.step(action)
            reward_sum += reward
            speed = env.k.vehicle.get_speed(rl_id)
            speed_list.append(next_state[0] * 15)

            next_state = running_state(next_state)

            total_step += 1

            mask = 1
            if done:
                mask = 0

            memory.push(state, np.array([rl_action]), mask, next_state, reward)

            if args.render:
                env.render()
            if done:
                break

            state = next_state
        num_steps += (t + 1)
        num_episodes += 1

        reward_batch += [reward_sum]
        speed_mean_batch += [np.mean(speed_list)]
        speed_std_batch += [np.std(speed_list)]

    if np.mean(speed_mean_batch) < 0:
        speed_mean_batch = [0]
        speed_std_batch = [0]
    writer.add_scalar('Reward/reward_batch', np.mean(reward_batch), total_step)
    writer.add_scalar('Speed/speed_mean_batch', np.mean(speed_mean_batch), total_step)
    writer.add_scalar('Speed/speed_std_batch', np.mean(speed_std_batch), total_step)

    batch = memory.sample()
    update_params(batch)

    torch.save(policy_net, checkpoints_path + '/policy' + str(i_episode) + '.pth')
    torch.save(value_net, checkpoints_path + '/value' + str(i_episode) + '.pth')

    if i_episode % args.log_interval == 0:
        print('Episode {}\tLast reward: {}\tAverage reward {:.2f}'.format(
            i_episode, reward_sum, np.mean(reward_batch)))
    i_episode += 1

    # Train FakeEnv
    if np.mean(reward_batch) < args.reward_threshold:
        print(">>>>>>>>>>>>>>> Interact With Fake Env <<<<<<<<<<<<<<<<<<<<")
        buffer_train = Memory()
        num_steps = 0
        while num_steps < args.batch_size:
            state = env.reset()
            rl_id = env.k.vehicle.get_rl_ids()[0]
            for t in range(10000):  # Don't infinite loop while learning
                action = np.random.uniform(-1, 1)
                next_state, reward, done, _ = env.step(action)

                mask = 1
                if done:
                    mask = 0

                if np.abs(state[2]) < 1e-4:
                    pass
                else:
                    buffer_train.push(state, np.array([action]), mask, next_state, reward)

                if done:
                    break

                state = next_state
            num_steps += (t - 1)

        print(">>>>>>>>>>>>>>> Train Fake Env <<<<<<<<<<<<<<<<<<<<")
        batch_train = buffer_train.sample()
        loss_state, loss_reward = fake_env.train_model(batch_train)
        print("state  loss is ", loss_state)
        print("reward loss is ", loss_reward)

        # Train Agent with Fake Env
        print(">>>>>>>>>>>>>>> Train Agent <<<<<<<<<<<<<<<<<<<<")
        if loss_state < args.loss_state_threshold and loss_reward < args.reward_state_threshold:
            for i in range(5):
                memory_fake = Memory()
                state = env.reset()
                state_r = running_state(state)
                rl_id = env.k.vehicle.get_rl_ids()[0]
                k_star = min(args.k_max, int(args.rollout_sensitivity / loss_state))
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> rollout steps {}".format(k_star))
                for t in range(k_star):
                    action = select_action(state_r)
                    rl_action = action.data[0].numpy()
                    phy_action = env.k.vehicle.get_acc_controller(rl_id).get_accel(env)
                    action = rl_action + phy_action

                    next_state, reward = fake_env.step(state, action)

                    next_state_r = running_state(next_state)

                    mask = 1
                    if t == k_star-1:
                        mask = 0

                    memory_fake.push(state_r, np.array([rl_action]), mask, next_state_r, reward)

                    state = next_state
                    state_r = running_state(state)
                batch_fake = memory_fake.sample()
                update_params(batch_fake)

writer.close()
