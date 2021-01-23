#!/usr/bin/env python3

import os
import random
import time
import numpy as np
from skimage.transform import resize

import gym

from mfec.agent import RandomAgent
from utils import Utils
from sklearn.decomposition import PCA, NMF

import os.path
import pickle

EPOCHS = 1
FRAMES_PER_EPOCH = 10000
SEED = 42

ACTION_BUFFER_SIZE = 100000
K = 11
DISCOUNT = 1
EPSILON = 0.005

FRAMESKIP = 4  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25

SCALE_HEIGHT = 84
SCALE_WIDTH = 84
STATE_DIMENSION = 64


def train_random_pca(environment, agent_path):
    random.seed(SEED)

    # Initialize utils, environment and agent
    utils = Utils(agent_path, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(environment)

    try:
        env.env.frameskip = FRAMESKIP
        env.env.ale.setFloat("repeat_action_probability", REPEAT_ACTION_PROB)
        pca_agent = RandomAgent(
                ACTION_BUFFER_SIZE,
                K,
                DISCOUNT,
                EPSILON,
                SCALE_HEIGHT,
                SCALE_WIDTH,
                STATE_DIMENSION,
                range(env.action_space.n),
                SEED,
            )
        pca = train_pca(pca_agent, env, utils, agent_path)

    finally:
        utils.close()
        env.close()
    return pca

def save(results_dir, data):
        with open(os.path.join(results_dir, "agent_pca.pkl"), "wb") as file:
            pickle.dump(data, file, 2)


def load(path):
    path = path.split('.')[0] + "_pca.pkl"
    with open(path, "rb") as file:
        return pickle.load(file)


def train_pca(agent, env, utils, path):
    frames_left = 0
    observation_x = []
    for epoch in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            observations, episode_frames, episode_reward = run_episode(agent, env)
            observation_x.extend(observations)
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
    
    return train_pca_embedding(observation_x, path)

def run_episode(agent, env):
    observations = []
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()
    done = False
    while not done:
        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)
        observations.append(np.array(observation))
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return observations, episode_frames, episode_reward

def train_pca_embedding(observations, path):
    pca = PCA(STATE_DIMENSION)
    print("pca is learning")
    for i in range(len(observations)):
        observations[i] = observations[i].flatten()
    pca = pca.fit(observations)
    save(path, observations)
    print("pca learned")
    return pca

def load_pca(path):
    observations = load(path)
    pca = PCA(STATE_DIMENSION)
    pca = pca.fit(observations)
    return pca

if __name__ == "__main__":
    ENVIRONMENT = "Qbert-v0"
    train_random_pca(ENVIRONMENT, '.')

