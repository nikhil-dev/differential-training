import tensorflow as tf

import tensorflow.contrib.eager as tfe

class Actor(tfe.Network):
  def __init__(self, nb_actions, nb_obs, layer_norm=True):
	super(Actor, self).__init__(name='actor_scope')
	self.nb_actions = nb_actions
	self.layer_norm = layer_norm
	self._input_shape = [-1, nb_obs]

	self.fc1 = self.track_layer(tf.layers.Dense(64))
	self.fc2 = self.track_layer(tf.layers.Dense(64))
	self.batch_norm = self.track_layer(tf.layers.BatchNormalization())
	self.final_layer = self.track_layer(tf.layers.Dense(self.nb_actions))

  def call(self, obs):
  	x = tf.reshape(obs, self._input_shape)
	x = self.fc1(x)
	if self.layer_norm:
		x = self.batch_norm(x)
	x = tf.nn.relu(x)

	x = self.fc2(x)
	if self.layer_norm:
		x = self.batch_norm(x)
	x = tf.nn.relu(x)

	x = self.final_layer(x)
	return tf.nn.tanh(x)


class Critic(tfe.Network):
  def __init__(self, nb_actions, nb_obs, layer_norm=True):
	super(Critic, self).__init__(name='critic_scope')
	self.layer_norm = layer_norm

	self.fc1 = self.track_layer(tf.layers.Dense(64))
	self.fc2 = self.track_layer(tf.layers.Dense(64))
	self.batch_norm = self.track_layer(tf.layers.BatchNormalization())
	self.final_layer = self.track_layer(tf.layers.Dense(1))
	self._input_action_shape = [-1, nb_actions]
	self._input_obs_shape = [-1, nb_obs]

  def call(self, obs, actions):
	obs = tf.reshape(obs, self._input_obs_shape)
	actions = tf.reshape(actions, self._input_action_shape)

	x = obs
	x = self.fc1(x)
	if self.layer_norm:
		x = self.batch_norm(x)
	x = tf.nn.relu(x)

	x = tf.concat([x, actions], axis=-1)
	x = self.fc2(x)
	if self.layer_norm:
		x = self.batch_norm(x)
	x = tf.nn.relu(x)

	return self.final_layer(x)


class DifferenceCritic(tfe.Network):
  def __init__(self, nb_actions, nb_obs, layer_norm=True):
	super(DifferenceCritic, self).__init__(name='difference_critic_scope')
	self.layer_norm = layer_norm

	self.fc1 = self.track_layer(tf.layers.Dense(64))
	self.fc2 = self.track_layer(tf.layers.Dense(64))
	self.batch_norm = self.track_layer(tf.layers.BatchNormalization())
	self.final_layer = self.track_layer(tf.layers.Dense(1))
	self._input_action_shape = [-1, nb_actions]
	self._input_obs_shape = [-1, nb_obs]

  def call(self, obs1, action1, obs2, action2):
	obs1 = tf.reshape(obs1, self._input_obs_shape)
	obs2 = tf.reshape(obs2, self._input_obs_shape)
	action1 = tf.reshape(action1, self._input_action_shape)
	action2 = tf.reshape(action2, self._input_action_shape)

	x = tf.concat([obs1, action1, obs2, action2], axis=-1)

	x = self.fc1(x)
	if self.layer_norm:
		x = self.batch_norm(x)
	x = tf.nn.relu(x)

	x = self.fc2(x)
	if self.layer_norm:
	  x = self.batch_norm(x)
	x = tf.nn.relu(x)

	return self.final_layer(x)
