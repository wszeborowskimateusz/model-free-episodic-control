# coding:utf-8

# INITIAL FILE REFERENCE 
# https://github.com/tokb23/dqn/blob/261dce2e1b43b9a60892b06be2c0b4e489cb134e/dqn.py

from keras.layers import Conv2D, Flatten, Dense
from keras.models import Sequential
from skimage.transform import resize
from skimage.color import rgb2gray
from collections import deque
import os
import random
import numpy as np

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


FRAME_WIDTH = 84  # Resized frame width
FRAME_HEIGHT = 84  # Resized frame height
NUM_EPISODES = 12000  # Number of episodes the agent plays
STATE_LENGTH = 4  # Number of most recent frames to produce the input to the network
GAMMA = 0.99  # Discount factor
# Number of steps over which the initial value of epsilon is linearly annealed to its final value
EXPLORATION_STEPS = 1000000
INITIAL_EPSILON = 1.0  # Initial value of epsilon in epsilon-greedy
FINAL_EPSILON = 0.1  # Final value of epsilon in epsilon-greedy
# Number of steps to populate the replay memory before training starts
INITIAL_REPLAY_SIZE = 20000
NUM_REPLAY_MEMORY = 400000  # Number of replay memory the agent uses for training
BATCH_SIZE = 32  # Mini batch size
# The frequency with which the target network is updated
TARGET_UPDATE_INTERVAL = 10000
TRAIN_INTERVAL = 4  # The agent selects 4 actions between successive updates
LEARNING_RATE = 0.00025  # Learning rate used by RMSProp
MOMENTUM = 0.95  # Momentum used by RMSProp
# Constant added to the squared gradient in the denominator of the RMSProp update
MIN_GRAD = 0.01


class DQNAgent():
	def __init__(self, num_actions):
		self.num_actions = num_actions
		self.epsilon = INITIAL_EPSILON
		self.epsilon_step = (INITIAL_EPSILON - FINAL_EPSILON) / EXPLORATION_STEPS
		self.t = 0

		# Create replay memory
		self.replay_memory = deque()

		# Create q network
		self.s, self.q_values, q_network = self._build_network()
		q_network_weights = q_network.trainable_weights

		# Create target network
		self.st, self.target_q_values, target_network = self._build_network()
		target_network_weights = target_network.trainable_weights

		# Define target network update operation
		self.update_target_network = [target_network_weights[i].assign(
		    q_network_weights[i]) for i in range(len(target_network_weights))]

		# Define loss and gradient update operation
		self.a, self.y, self.loss, self.grads_update = self._build_training_op(
		    q_network_weights)

		self.sess = tf.InteractiveSession()
		self.saver = tf.train.Saver(q_network_weights)

		self.sess.run(tf.global_variables_initializer())

		# Initialize target network
		self.sess.run(self.update_target_network)

	def _build_network(self):
		model = Sequential()
		model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=(
		    FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH)))  # channel last
		model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
		model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
		model.add(Flatten())
		model.add(Dense(512, activation='relu'))
		model.add(Dense(self.num_actions))

		s = tf.placeholder(
		    tf.float32, [None, FRAME_WIDTH, FRAME_HEIGHT, STATE_LENGTH])
		q_values = model(s)

		return s, q_values, model

	def _build_training_op(self, q_network_weights):
		a = tf.placeholder(tf.int64, [None])
		y = tf.placeholder(tf.float32, [None])

		# Convert action to one hot vector
		a_one_hot = tf.one_hot(a, self.num_actions, 1.0, 0.0)
		q_value = tf.reduce_sum(tf.multiply(
		    self.q_values, a_one_hot), reduction_indices=1)

		# Clip the error, the loss is quadratic when the error is in (-1, 1), and linear outside of that region
		error = tf.abs(y - q_value)
		quadratic_part = tf.clip_by_value(error, 0.0, 1.0)
		linear_part = error - quadratic_part
		loss = tf.reduce_mean(0.5 * tf.square(quadratic_part) + linear_part)

		optimizer = tf.train.RMSPropOptimizer(
		    LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
		grads_update = optimizer.minimize(loss, var_list=q_network_weights)

		return a, y, loss, grads_update

	def get_initial_state(self, observation, last_observation):
		processed_observation = np.maximum(observation, last_observation)
		processed_observation = np.uint8(
		    resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
		state = [processed_observation for _ in range(STATE_LENGTH)]
		state = np.stack(state, axis=0)  # (4, 84, 84)
		state = np.rollaxis(np.rollaxis(state, 1, 0), 2, 1)
		return state

	def choose_action(self, state):
		if random.random() <= self.epsilon or self.t < INITIAL_REPLAY_SIZE:
			action = random.randrange(self.num_actions)
		else:
			action = np.argmax(self.q_values.eval(feed_dict={self.s: [np.float32(state / 255.0)]}))

		# Anneal epsilon linearly over time
		if self.epsilon > FINAL_EPSILON and self.t >= INITIAL_REPLAY_SIZE:
			self.epsilon -= self.epsilon_step

		return action

	def run(self, state, action, reward, terminal, observation):
		# append observation in state and update to keep the STATE_LENGTH at 4
		next_state = np.append(state[:, :, 1:], observation, axis=2)  

		# Clip all positive rewards at 1 and all negative rewards at -1, leaving 0 rewards unchanged
		reward = np.clip(reward, -1, 1)

		# Store transition in replay memory
		self.replay_memory.append((state, action, reward, next_state, terminal))
		if len(self.replay_memory) > NUM_REPLAY_MEMORY:
			self.replay_memory.popleft()

		if self.t >= INITIAL_REPLAY_SIZE:
			# Train network
			if self.t % TRAIN_INTERVAL == 0:
				self._train_network()

			# Update target network
			if self.t % TARGET_UPDATE_INTERVAL == 0:
				self.sess.run(self.update_target_network)

		self.t += 1

		return next_state

	def save(self, results_dir):
		self.saver.save(self.sess, os.path.join(results_dir, "agent.pkl"), global_step=self.t)

	def _train_network(self):
		state_batch = []
		action_batch = []
		reward_batch = []
		next_state_batch = []
		terminal_batch = []
		y_batch = []

		# Sample random minibatch of transition from replay memory
		minibatch = random.sample(self.replay_memory, BATCH_SIZE)
		for data in minibatch:
			state_batch.append(data[0])
			action_batch.append(data[1])
			reward_batch.append(data[2])
			next_state_batch.append(data[3])
			terminal_batch.append(data[4])

		# Convert True to 1, False to 0
		terminal_batch = np.array(terminal_batch) + 0

		target_q_values_batch = self.target_q_values.eval(feed_dict={self.st: np.float32(np.array(next_state_batch) / 255.0)})
		y_batch = reward_batch + (1 - terminal_batch) * GAMMA * np.max(target_q_values_batch, axis=1)

		loss, _ = self.sess.run([self.loss, self.grads_update], feed_dict={
			self.s: np.float32(np.array(state_batch) / 255.0),
			self.a: action_batch,
			self.y: y_batch
		})

	def load(self, path):
		checkpoint = tf.train.get_checkpoint_state(path)
		if checkpoint and checkpoint.model_checkpoint_path:
			self.saver.restore(self.sess, checkpoint.model_checkpoint_path)


def preprocess(observation, last_observation):
	processed_observation = np.maximum(observation, last_observation)
	processed_observation = np.uint8(resize(rgb2gray(processed_observation), (FRAME_WIDTH, FRAME_HEIGHT)) * 255)
		
	state = np.reshape(processed_observation, (1, FRAME_WIDTH, FRAME_HEIGHT))
	state = np.rollaxis(np.rollaxis(state, 1,0), 2,1)
	return state