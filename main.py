import argparse
import time
import os

import gym
import tensorflow as tf
import tensorflow.contrib.eager as tfe

from models import Actor, Critic, DifferenceCritic
from memory import Memory
import training

def run(env_id, seed, batch_norm, evaluation, **kwargs):
	# Create environments
	env = gym.make(env_id)

	if evaluation:
		eval_env = gym.make(env_id)
	else:
		eval_env = None

	nb_actions = env.action_space.shape[-1]
	nb_obs = env.observation_space.shape[-1]

	# Configure actor, critic, difference critic and memory
	memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
	critic = Critic(nb_actions=nb_actions, nb_obs=nb_obs, layer_norm=batch_norm)
	actor = Actor(nb_actions, nb_obs, layer_norm=batch_norm)
	difference_critic = DifferenceCritic(nb_actions=nb_actions, nb_obs=nb_obs, layer_norm=batch_norm)

	# Seed to make things reproducible
	env.seed(seed)
	if eval_env is not None:
		eval_env.seed(seed)

	start_time = time.time()

	# The main training loop
	training.train(env=env, eval_env=eval_env, actor=actor, critic=critic, difference_critic=difference_critic, memory=memory, **kwargs)

	# Clean up
	env.close()
	if eval_env is not None:
		eval_env.close()
	print ('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
	parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('--env-id', type=str, default='HalfCheetah-v1')
	parser.add_argument('--render-eval', action='store_true')
	parser.add_argument('--use-difference-critic', action='store_true')
	parser.add_argument('--disable-batch-norm', action='store_true')
	parser.add_argument('--render', action='store_true')
	parser.add_argument('--normalize-returns', action='store_true')
	parser.add_argument('--normalize-observations', action='store_true')
	parser.add_argument('--seed', help='RNG seed', type=int, default=0)
	parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
	parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
	parser.add_argument('--actor-lr', type=float, default=1e-4)
	parser.add_argument('--critic-lr', type=float, default=1e-3)
	parser.add_argument('--difference-critic-lr', type=float, default=1e-4)
	parser.add_argument('--gamma', type=float, default=0.99)
	parser.add_argument('--reward-scale', type=float, default=1.)
	parser.add_argument('--clip-norm', type=float, default=None)
	parser.add_argument('--nb-epochs', type=int, default=500)  # with default settings, perform 1M steps total
	parser.add_argument('--nb-epoch-cycles', type=int, default=20)
	parser.add_argument('--nb-train-steps', type=int, default=50)  # per epoch cycle and MPI worker
	parser.add_argument('--nb-eval-steps', type=int, default=100)  # per epoch cycle and MPI worker
	parser.add_argument('--nb-rollout-steps', type=int, default=100)  # per epoch cycle and MPI worker
	parser.add_argument('--num-timesteps', type=int, default=None)
	parser.add_argument('--evaluation', action='store_true')
	parser.add_argument('--checkpoint-dir', default='/tmp/ddpg/hello-world/4')
	args = parser.parse_args()
	
	# we don't directly specify timesteps for this script, so make sure that if we do specify them
	# they agree with the other parameters
	if args.num_timesteps is not None:
		assert(args.num_timesteps == args.nb_epochs * args.nb_epoch_cycles * args.nb_rollout_steps)
	dict_args = vars(args)
	del dict_args['num_timesteps']

	# Set batch norm to true by default. This is a hack around parser.add_argument() being default false. 
	dict_args['batch_norm'] = True
	if dict_args['disable_batch_norm']:
		dict_args['batch_norm'] = False
	del dict_args['disable_batch_norm']

	return dict_args


if __name__ == '__main__':
	tfe.enable_eager_execution()
	args = parse_args()
	run(**args)
