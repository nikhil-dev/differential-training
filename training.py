import os
import time
from collections import deque
import pickle
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from ddpg import DDPG

def apply_noise(tensor, noise_multiple):
	return tensor
	noise = np.random.normal(np.zeros(tensor.shape, dtype=np.float32), scale=np.abs(tensor)*noise_multiple)
	noise = noise.astype('float32')
	return tensor + noise


def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, actor, critic, difference_critic,
	normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, difference_critic_lr,
	gamma, clip_norm, nb_train_steps, nb_rollout_steps, nb_eval_steps, batch_size, memory, checkpoint_dir,
	observation_noise_multiple, action_noise_multiple, use_difference_critic, tau=0.01, eval_env=None):

	assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
	agent = DDPG(actor, critic, difference_critic, memory, env.observation_space.shape, env.action_space.shape,
		use_difference_critic=use_difference_critic, gamma=gamma, tau=tau, normalize_returns=normalize_returns,
		normalize_observations=normalize_observations, batch_size=batch_size, critic_l2_reg=critic_l2_reg, actor_lr=actor_lr,
		critic_lr=critic_lr, difference_critic_lr=difference_critic_lr, clip_norm=clip_norm, reward_scale=reward_scale,
		action_range=(env.action_space.low, env.action_space.high))
	
	print('Using agent with the following configuration:')
	print(str(agent.__dict__.items()))
	max_action = env.action_space.high
	print('scaling actions by {} before executing in env'.format(max_action))

	obs = env.reset()
	# inject noise to simulate model ensemble
	obs = apply_noise(obs, observation_noise_multiple)

	if eval_env is not None:
		eval_obs = eval_env.reset()

	done = False
	episode_reward = 0.
	episode_step = 0
	episodes = 0
	time_step = 0
	eval_episode_rewards_history = deque(maxlen=100)
	episode_rewards_history = deque(maxlen=100)

	tf.gfile.MakeDirs(checkpoint_dir)

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
				action = apply_noise(action, action_noise_multiple)
				action = tf.clip_by_value(action, env.action_space.low, env.action_space.high)
				new_obs, r, done, info = env.step(max_action * action)
				# inject noise
				new_obs = apply_noise(new_obs, observation_noise_multiple)

				time_step += 1
				if render:
					env.render()
				episode_reward += r
				episode_step += 1

				# Add to replay buffer.
				agent.store_transition(obs, action, r, new_obs, done)
				obs = new_obs

				if done:
					# Episode done.
					episode_rewards_history.append(episode_reward)
					episode_reward = 0.
					episode_step = 0
					episodes += 1

					obs = env.reset()
					obs = apply_noise(obs, observation_noise_multiple)


			# Train.
			checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
			with tfe.restore_variables_on_create(tf.train.latest_checkpoint(checkpoint_dir)):
				for t_train in range(nb_train_steps):
					global_step = tf.train.get_or_create_global_step()
					agent.train(global_step)
					print '...'
			tfe.Saver(agent.get_variables_to_save()).save(checkpoint_prefix, 
				global_step=tf.train.get_or_create_global_step())

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
						print "eval episode reward: " + str(eval_episode_reward)
						eval_obs = eval_env.reset()
						eval_episode_rewards.append(eval_episode_reward)
						eval_episode_rewards_history.append(eval_episode_reward)
						eval_episode_reward = 0.

