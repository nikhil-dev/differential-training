import os
import time
from collections import deque
import pickle
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ddpg import DDPG


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic, difference_critic,
	normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, difference_critic_lr,
	gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, checkpoint_dir,
	use_difference_critic=False, tau=0.01, eval_env=None):

	assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
	agent = DDPG(actor, critic, difference_critic, memory, env.observation_space.shape, env.action_space.shape,
		use_difference_critic=use_difference_critic, gamma=gamma, tau=tau, normalize_returns=normalize_returns, 
		normalize_observations=normalize_observations, batch_size=batch_size, critic_l2_reg=critic_l2_reg, actor_lr=actor_lr, 
		critic_lr=critic_lr, difference_critic_lr=difference_critic_lr, clip_norm=clip_norm, reward_scale=reward_scale)
	print('Using agent with the following configuration:')
	print(str(agent.__dict__.items()))

	max_action = env.action_space.high
	print('scaling actions by {} before executing in env'.format(max_action))

	step = 0
	episode = 0
	eval_episode_rewards_history = deque(maxlen=100)
	episode_rewards_history = deque(maxlen=100)

	obs = env.reset()
	if eval_env is not None:
		eval_obs = eval_env.reset()

	done = False
	episode_reward = 0.
	episode_step = 0
	episodes = 0
	t = 0

	epoch = 0
	start_time = time.time()

	epoch_episode_rewards = []
	epoch_episode_steps = []
	epoch_episode_eval_rewards = []
	epoch_episode_eval_steps = []
	epoch_start_time = time.time()
	epoch_actions = []
	epoch_qs = []
	epoch_episodes = 0
	for epoch in range(nb_epochs):
		for cycle in range(nb_epoch_cycles):
			for t_rollout in range(nb_rollout_steps):
				# Predict next action.
				action, q = agent.pi(obs)
				assert action.shape == env.action_space.shape

				# Execute next action.
				if render:
					env.render()
				# scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
				new_obs, r, done, info = env.step(max_action * action)
				t += 1
				if render:
					env.render()
				episode_reward += r
				episode_step += 1

				# Book-keeping.
				epoch_actions.append(action)
				epoch_qs.append(q) # this is predicted Q
				agent.store_transition(obs, action, r, new_obs, done)
				obs = new_obs

				if done:
					# Episode done.
					epoch_episode_rewards.append(episode_reward)
					episode_rewards_history.append(episode_reward)
					epoch_episode_steps.append(episode_step)
					episode_reward = 0.
					episode_step = 0
					epoch_episodes += 1
					episodes += 1

					obs = env.reset()

			# Train.
			checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
			for t_train in range(nb_train_steps):
				# with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_dir)):
				global_step = tf.train.get_or_create_global_step()
				agent.train(global_step)
				print '...'
				# tf.gfile.MakeDirs(checkpoint_dir)
				# tfe.Saver(agent.get_variables_to_save(global_step)).save(checkpoint_prefix, global_step=global_step)
				# sys.exit()

			# Evaluate.
			eval_episode_rewards = []
			eval_qs = []
			if eval_env is not None:
				eval_episode_reward = 0.
				for t_rollout in range(nb_eval_steps):
					eval_action, eval_q = agent.pi(eval_obs)
					# scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
					eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)
					if render_eval:
						eval_env.render()
					eval_episode_reward += eval_r

					eval_qs.append(eval_q)
					if eval_done:
						eval_obs = eval_env.reset()
						eval_episode_rewards.append(eval_episode_reward)
						eval_episode_rewards_history.append(eval_episode_reward)
						eval_episode_reward = 0.
