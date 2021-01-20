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


  # More games at: https://gym.openai.com/envs/#atari
AGENT_PATH = None#"agents/Qbert-v0_1542210528/agent.pkl"
RENDER = False #True
RENDER_SPEED = 0.04

EPOCHS = 2
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


def train_random_pca(environment):

    random.seed(SEED)

    # Create agent-directory
    execution_time = str(round(time.time()))
    agent_dir = os.path.join("agents", environment + "_" + execution_time)
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
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
        pca = train_pca(pca_agent, agent_dir, env, utils)

    finally:
        utils.close()
        env.close()
    return pca


def train_pca(agent, agent_dir, env, utils):
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
        agent.save(agent_dir)
    
    return train_pca_embedding(observation_x)

def run_episode(agent, env):
    observations = []
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()
    done = False
    while not done:

        if RENDER:
            env.render()
            time.sleep(RENDER_SPEED)

        action = agent.choose_action(observation)
        observation, reward, done, _ = env.step(action)
        observations.append(np.array(observation))
        agent.receive_reward(reward)

        episode_reward += reward
        episode_frames += FRAMESKIP

    agent.train()
    return observations, episode_frames, episode_reward

def train_pca_embedding(observations):
    pca = PCA(STATE_DIMENSION)
    print("pca is learning")
    for i in range(len(observations)):
        observations[i] = observations[i].flatten()
    pca = pca.fit(observations)
    print("pca learned")
    return pca


if __name__ == "__main__":
    ENVIRONMENT = "Qbert-v0"
    train_random_pca(ENVIRONMENT)

