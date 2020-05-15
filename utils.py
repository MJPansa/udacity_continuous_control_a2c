import torch as T
import torch.optim as optim
import torch.nn.modules as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.nn.functional as F
import torch.distributions as dist
import numpy as np
import sys


class ActorCritic(nn.Module):
    def __init__(self, n_states, n_actions, n_hidden, lr):
        super(ActorCritic, self).__init__()

        self.input = nn.Linear(n_states, n_hidden)
        self.hidden_1 = nn.Linear(n_hidden, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.out_actor_sigma = nn.Linear(n_hidden, n_actions)
        self.out_actor_mu = nn.Linear(n_hidden, n_actions)
        self.out_critic = nn.Sequential(nn.Linear(n_hidden, n_hidden),
                                        nn.ReLU(), nn.Linear(n_hidden, 1))

        #self.bn_1 = nn.BatchNorm1d(n_hidden)
        #self.bn_2 = nn.BatchNorm1d(n_hidden)

        self.optimizer = optim.SGD(self.parameters(), lr=lr)

    def forward(self, x):
        if not self.training:
            x = x.float()

        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))

        mus = F.tanh(self.out_actor_mu(x))
        sigmas = F.softplus(self.out_actor_sigma(x))
        sigmas = T.clamp(sigmas, 5e-4, 2)
        value = self.out_critic(x.detach())

        return mus, sigmas, value.squeeze()


def play_episode(env, brain_name, model, device, n_steps):
    states_list = []
    rewards_list = []
    actions_list = []
    values_list = []
    model.eval()

    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    t = 0
    while t < n_steps:
        t += 1
        mus, sigmas, values = model(T.from_numpy(states).to(device))
        distribution = dist.Normal(mus, sigmas)
        actions = distribution.sample()
        actions = T.clamp(actions, -1, 1)
        env_info = env.step(actions.squeeze().detach().cpu().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        values_list.append(values)
        states_list.append(T.from_numpy(states).to(device))
        rewards_list.append(rewards)
        actions_list.append(actions.squeeze())
        states = next_states.copy()
        if np.any(dones):
            break
    return states_list, rewards_list, actions_list, values_list


def calculate_g(rewards_list, values_list, gamma):
    rewards = np.flip(np.stack(rewards_list), axis=0)
    last_states = values_list[-1].detach().cpu().numpy()
    for i in range(rewards.shape[0]):
        if i == 0:
            rewards[i, :] = rewards[i, :] + gamma * last_states
        else:
            rewards[i, :] = rewards[i, :] + gamma * rewards[i - 1, :]
    return np.flip(rewards)


def train_on_episode(model, states_list, action_list, rewards_list, lr, critic_weight, device, entropy_beta):
    model.train()

    actions_tensor = T.stack(action_list, dim=0)
    states = T.stack(states_list)
    states = T.stack([states[:, n, :] for n in range(20)], dim=0)
    states = states.view(-1, states.size(-1))
    actions = T.stack([actions_tensor[:, n, :] for n in range(20)], dim=0)
    actions = actions.view(-1, actions.size(-1))
    rewards = T.stack([T.Tensor(rewards_list[:, n]) for n in range(20)], dim=0).to(device)
    rewards = rewards.view(-1)

    #rewards = F.normalize(rewards, dim=0)

    mus, sigmas, state_values = model(states.float())
    gaussian = dist.Normal(mus, sigmas)
    logprobs = gaussian.log_prob(actions.float().to(device)).squeeze().float().mean(1)
    entropy = gaussian.entropy().squeeze().float()
    entropy_loss = (-1 * entropy.mean(1) * entropy_beta).sum()
    actor_loss = (-1 * (logprobs * (rewards.float() - state_values.squeeze().detach().float()))).sum() + entropy_loss
    critic_loss = (T.pow((rewards - state_values.squeeze()), 2)).sum()
    loss = actor_loss + (critic_weight * critic_loss)

    model.optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), 5.)
    model.optimizer.step()

    mean_sigma = sigmas.mean()

    return [x.detach().cpu().item() for x in [actor_loss, critic_loss, loss, entropy_loss, mean_sigma]]


def play_test_episode(env, brain_name, model, device):
    rewards_list = []
    model.eval()
    done = False
    env_info = env.reset(train_mode=True)[brain_name]
    states = env_info.vector_observations
    while not done:
        mus, sigmas, values = model(T.from_numpy(states).to(device))
        distribution = dist.Normal(mus, sigmas)
        actions = distribution.sample()
        actions = T.clamp(actions, -1, 1)

        env_info = env.step(actions.squeeze().detach().cpu().numpy())[brain_name]
        next_states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        rewards_list.append(np.mean(rewards))
        states = next_states.copy()
        if np.any(dones):
            done = True
    return sum(rewards_list)
