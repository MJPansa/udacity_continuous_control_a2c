import numpy as np
import torch as T
from unityagents import UnityEnvironment
from utils import ActorCritic, play_episode, calculate_g, train_on_episode, play_test_episode
import wandb

wandb.init(project="actor_critic_continuous")

env = UnityEnvironment(
    file_name='Reacher_Linux/Reacher.x86_64', no_graphics=True)

brain_name = env.brain_names[0]
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]
states = env_info.vector_observations

N_STATES = states.shape[1]
N_ACTIONS = brain.vector_action_space_size
N_HIDDEN = 128
GAMMA = 0.99
LR = 5e-4
CRITIC_WEIGHT = 1
DEVICE = 'cuda:1'
ENTROPY_BETA = 0.0003
N_STEPS = 10


model = ActorCritic(n_states=N_STATES, n_actions=N_ACTIONS, n_hidden=N_HIDDEN, lr=LR)
model = model.to(DEVICE)

#wandb.watch(model)

run = 1
rewards_total = []

while True:
    states_list, rewards_list, actions_list, values_list = play_episode(env, brain_name, model, DEVICE, N_STEPS)
    G_list = calculate_g(rewards_list, values_list, GAMMA)
    actor_loss, critic_loss, loss, entropy_loss, mean_sigma = train_on_episode(model,
                                                                               states_list, actions_list, G_list,
                                                                               LR, CRITIC_WEIGHT, DEVICE, ENTROPY_BETA)

    run += 1

    if run % 5 == 0:
        mean_rewards = play_test_episode(env, brain_name, model, DEVICE)
        wandb.log({'rewards': mean_rewards,
                   'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                   'loss': loss,
                   'entropy_loss': entropy_loss,
                   'mean_sigma': mean_sigma})
        rewards_total.append(mean_rewards)
        print(f'Finished run {run}\nmean-rewards: {np.mean(rewards_total[-10:]):.2f}\nLast test-run: {mean_rewards:.2f}\n')

        if np.mean(rewards_total[-10:]) > 31.:
            print(f'SOLVED AFTER {run} n-step updates')
            break

    if run % 1000 == 0:
        T.save(model, f'a2c_multi_reacher_run_{run}_{mean_rewards:.2}.h5')
        T.save(model.state_dict(), f'a2c_multi_reacher_run_{run}_{mean_rewards:.2}_dict.h5')

T.save(model, f'a2c_multi_reacher_run_{run}_solved.h5')
T.save(model.state_dict(), f'a2c_multi_reacher_run_{run}_solved_dict.h5')

