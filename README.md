# Difference Critic

## Motivation

The goal is to train a RL system that learns a difference of value functions in order to perform effectively under simulation and approximation errors. In other words, there is mis-match between simulated and target domains. This is a OpenAI Request for Research problem ["Difference of Value Functions"](https://openai.com/requests-for-research/#difference-of-value-functions).


## What's New

The main idea idea comes from a 1997 paper [Differential Training of Rollout Policies, by Bertsekas](http://web.mit.edu/dimitrib/www/Diftrain.pdf). The paper introduces a technique called differential training and argues that under simulation and approximation error, learning a difference of value functions  can do better than learning vanilla value functions.

Instead of learning a difference of value function as suggested by Bertsekas, in this work I introduce a variant of DDPG ([Deep Deterministic Policy Gradients](https://arxiv.org/abs/1509.02971)), which instead of learning a ```Q(state, action)``` function, learns a difference of Q function ```Q(state1, action1, state2, action2)``` which approximates the difference of expected Q-values between two ```state, action``` pairs under the current policy. We use the gradient from this function to train the policy network in DDPG. 

## Implementation Details

The mismatch between simulated and target domains is modeled using Mujoco agents with varying torso masses, similar to [EPOpt](https://arxiv.org/abs/1610.01283). As in EPOpt, we train on a ensemble of robot models.

We use the Mujoco physics simulator for training on the ```HalfCheetah-v1``` environment.

We use a Tensorflow Eager adaptation of [OpenAI Baselines](https://github.com/openai/baselines) for [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) as the baseline.

This model has been ported to Tensorflow Eager, which gives us a better Pythonic expression of the model (define-by-run as opposed to define-and-run) and makes it easier to debug in many cases.

## Instructions to for installation

1. Install [OpenAI Gym and Mujoco](https://github.com/openai/gym) (needs a software license).
2. Install Tensorflow from the nightly build (we need nightly builds for TF Eager unless you have >=1.5)
3. Install pybullet
4. Install numpy

## Future work

Apply the concept of differential training to other Deep RL methods and see if this gives us benefits in the presence of simulation error.

