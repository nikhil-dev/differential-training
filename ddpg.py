import six

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.python.util import tf_inspect

def filter_grads_and_vars_by_scope(grads_and_vars, scope):
	filtered_grads_and_vars = []
	for grad, var in grads_and_vars:
		if scope in var.__dict__['_handle_name']:
			filtered_grads_and_vars.append((grad, var))
	return filtered_grads_and_vars


class DDPG(object):
	def __init__(self, actor, critic, difference_critic, memory, observation_shape, action_shape,
		action_range, use_difference_critic, gamma=0.99, tau=0.001, normalize_returns=False,
		batch_size=128, return_range=(-np.inf, np.inf), normalize_observations=True, reward_scale=1.,
		critic_l2_reg=0., actor_lr=1e-4, difference_critic_lr=1e-4, critic_lr=1e-3, clip_norm=None):

		# Parameters.
		self.gamma = gamma
		self.tau = tau
		self.memory = memory
		self.normalize_observations = normalize_observations
		self.normalize_returns = normalize_returns
		self.action_range = action_range
		self.return_range = return_range
		self.critic = critic
		self.actor = actor
		self.difference_critic = difference_critic
		self.actor_lr = actor_lr
		self.critic_lr = critic_lr
		self.clip_norm = clip_norm
		self.reward_scale = reward_scale
		self.batch_size = batch_size
		self.stats_sample = None
		self.critic_l2_reg = critic_l2_reg
		self.use_difference_critic = use_difference_critic

		self.critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=critic_lr, name='critic_optimizer')
		self.actor_optimizer = tf.train.AdamOptimizer(learning_rate=actor_lr, name='actor_optimizer')
		self.difference_critic_optimizer = tf.train.GradientDescentOptimizer(learning_rate=difference_critic_lr, name='difference_critic_optimizer')

	def critic_loss(self, obs, actions, target_Qs):
		return tf.reduce_mean(tf.square(self.critic(obs, actions) - target_Qs))

	def difference_critic_loss(self, obsAs, actionAs, obsBs, actionBs, target_diff_Qs):
		return tf.reduce_mean(tf.square(self.difference_critic(obsAs, actionAs, obsBs, actionBs) - target_diff_Qs))

	def actor_loss(self, obs):
		return -tf.reduce_mean(self.critic(obs, self.actor(obs)))

	def actor_loss_with_difference_critic(self, obs, baseline_obs, baseline_actions):
		return -tf.reduce_mean(self.difference_critic(obs, self.actor(obs), baseline_obs, baseline_actions))
	
	def train(self, global_step):
		if self.use_difference_critic:
			# Sample replay buffer for training
			batchA = self.memory.sample(batch_size=self.batch_size * 2)
			batchB = self.memory.sample(batch_size=self.batch_size * 2)

			# Prepare training data for critic
			actionA1s = self.actor(tf.convert_to_tensor(batchA['obs1']))
			actionB1s = self.actor(tf.convert_to_tensor(batchB['obs1']))
			diff_Qs = self.difference_critic(tf.convert_to_tensor(batchA['obs1']), actionA1s, tf.convert_to_tensor(batchB['obs1']), actionB1s)
			target_diff_Qs = batchA['rewards'] - batchB['rewards'] + self.gamma * diff_Qs
			target_diff_Qs = tf.clip_by_value(tf.convert_to_tensor(target_diff_Qs), 2*self.return_range[0], 2*self.return_range[1])
			target_diff_Qs = tf.constant(target_diff_Qs) # target_diff_Qs should be treated as a constant (like supervised learning labels)

			# Gradient descent step on critic
			diff_critic_grad_fn = tfe.implicit_gradients(self.difference_critic_loss)
			difference_critic_grad = diff_critic_grad_fn(tf.convert_to_tensor(batchA['obs0']),
				tf.convert_to_tensor(batchA['actions']), 
				tf.convert_to_tensor(batchB['obs0']),
				tf.convert_to_tensor(batchB['actions']),
				target_diff_Qs)
			difference_critic_grad = filter_grads_and_vars_by_scope(difference_critic_grad, 'difference_critic')
			self.difference_critic_optimizer.apply_gradients(difference_critic_grad, global_step=global_step)

			# Gradient descent step on actor
			actor_grad_fn = tfe.implicit_gradients(self.actor_loss_with_difference_critic)
			actor_grad = actor_grad_fn(tf.convert_to_tensor(batchA['obs0']),
				tf.convert_to_tensor(batchB['obs0']),
				tf.convert_to_tensor(batchB['actions'])
			)
			actor_grad = filter_grads_and_vars_by_scope(actor_grad, 'actor') # We only want to optimize the actor at this stage
			self.actor_optimizer.apply_gradients(actor_grad, global_step=global_step)
		else:
			# Sample replay buffer for training
			batch = self.memory.sample(batch_size=self.batch_size)

			# Prepare training data for critic
			action1s = self.actor(tf.convert_to_tensor(batch['obs1']))
			Qs = self.critic(tf.convert_to_tensor(batch['obs1']), action1s)
			terminals = batch['terminals1'].astype('float32')
			target_Qs = batch['rewards'] + (1. - terminals) * self.gamma * Qs
			target_Qs = tf.clip_by_value(tf.convert_to_tensor(target_Qs), self.return_range[0], self.return_range[1])
			target_Qs = tf.constant(target_Qs) # target_Qs should be treated as a constant (like supervised learning labels)

			# Gradient descent step on critic_loss
			critic_grad_fn = tfe.implicit_gradients(self.critic_loss)
			critic_grad = critic_grad_fn(tf.convert_to_tensor(batch['obs0']),
				tf.convert_to_tensor(batch['actions']),
				target_Qs)
			critic_grad = filter_grads_and_vars_by_scope(critic_grad, 'critic')
			self.critic_optimizer.apply_gradients(critic_grad, global_step=global_step)

			# Gradient descent step on actor
			actor_grad_fn = tfe.implicit_gradients(self.actor_loss)
			actor_grad = actor_grad_fn(tf.convert_to_tensor(batch['obs0']))
			actor_grad = filter_grads_and_vars_by_scope(actor_grad, 'actor')
			self.actor_optimizer.apply_gradients(actor_grad, global_step=global_step)

	def store_transition(self, obs0, action, reward, obs1, terminal1):
		reward *= self.reward_scale
		self.memory.append(obs0, action, reward, obs1, terminal1)

	def pi(self, obs):
		obs = tf.convert_to_tensor(obs)
		action = self.actor(obs)
		q = self.critic(obs, action)
		action = tf.reshape(action, [-1])
		action = tf.clip_by_value(action, self.action_range[0], self.action_range[1])
		return action, q

	def get_variables_to_save(self):
		global_step = tf.train.get_or_create_global_step()

		var_list = (
			self.actor.variables
			+ self.critic.variables
			+ self.difference_critic.variables
			+ self.actor_optimizer.variables()
			+ self.critic_optimizer.variables()
			+ [global_step])
		return (
			self.actor.variables
			+ self.critic.variables
			+ self.difference_critic.variables 
			+ self.actor_optimizer.variables()
			+ self.critic_optimizer.variables() 
			+ self.difference_critic_optimizer.variables() 
			+ [global_step])
