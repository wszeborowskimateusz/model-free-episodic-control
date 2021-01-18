#!/usr/bin/env python3

import os
import random
import time

import gym

from mfec.agent import MFECAgent
from utils import Utils
from dqn.agent import DQNAgent, preprocess
from vae_train import train_random_vae
from mfec.agent import RandomAgent


ENVIRONMENT = "MsPacman-v0"  # More games at: https://gym.openai.com/envs/#atari
AGENT_PATH = None#"agents/MFEC/MsPacman-v0_1609170206/agent.pkl"

# MFEC or DQN
ALGORITHM = 'MFEC'

RENDER = True
RENDER_SPEED = 0.04

EPOCHS = 11
EPOCH_DELAY = 3
FRAMES_PER_EPOCH = 100000
SEED = 42

ACTION_BUFFER_SIZE = 1000000
K = 11
DISCOUNT = 1
EPSILON = 0.005

FRAMESKIP = 4  # Default gym-setting is (2, 5)
REPEAT_ACTION_PROB = 0.0  # Default gym-setting is .25

SCALE_HEIGHT = 84
SCALE_WIDTH = 84
STATE_DIMENSION = 64

# Maximum number of "do nothing" actions to be performed by the agent at the start of an episode
NO_OP_STEPS = {"DQN": 30, "MFEC": 0}

def main():
    random.seed(SEED)

    # Create agent-directory
    execution_time = str(round(time.time()))

    agent_dir = os.path.join("agents", ALGORITHM, ENVIRONMENT + "_" + execution_time)
    os.makedirs(agent_dir)

    # Initialize utils, environment and agent
    utils = Utils(agent_dir, FRAMES_PER_EPOCH, EPOCHS * FRAMES_PER_EPOCH)
    env = gym.make(ENVIRONMENT)

    try:
        env.env.frameskip = FRAMESKIP
        env.env.ale.setFloat("repeat_action_probability", REPEAT_ACTION_PROB)
        if ALGORITHM == 'MFEC':
            if AGENT_PATH:
                agent = MFECAgent.load(AGENT_PATH)
            else:
                agent = MFECAgent(
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
        else:
            agent = DQNAgent(env.action_space.n)
            if AGENT_PATH:
                agent.load(AGENT_PATH)
        
        run_algorithm(agent, agent_dir, env, utils)

    finally:
        utils.close()
        env.close()


def run_algorithm(agent, agent_dir, env, utils):
    frames_left = 0
    observations = []
    vae = train_random_vae(ENVIRONMENT)
    for epoch in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:

            episode_observations, episode_frames, episode_reward = run_episode(agent, env, vae)
            observations.extend(episode_observations)
            #if epoch % EPOCH_DELAY == 0: dotrenowa

            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
        agent.save(agent_dir)


def run_episode(agent, env, vae):
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()

    no_op_steps = NO_OP_STEPS[ALGORITHM]
    if no_op_steps > 0:
        for _ in range(random.randint(1, no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing

    if ALGORITHM == 'DQN':
        state = agent.get_initial_state(observation, last_observation)

    done = False
    while not done:

        if RENDER:
            env.render()
            time.sleep(RENDER_SPEED)

        last_observation = observation

        if ALGORITHM == 'DQN':
            action = agent.choose_action(state)
        else:
            small_observation = vae.encoder.predict(observation)
            action = agent.choose_action(small_observation)

        observation, reward, done, _ = env.step(action)


        if ALGORITHM == 'MFEC':
            agent.receive_reward(reward)

        if ALGORITHM == 'DQN':
            processed_observation = preprocess(observation, last_observation)
            state = agent.run(state, action, reward, done, processed_observation)


        episode_reward += reward
        episode_frames += FRAMESKIP

    if ALGORITHM == 'MFEC':
        agent.train()
    return episode_frames, episode_reward
    

if __name__ == "__main__":
    main()
