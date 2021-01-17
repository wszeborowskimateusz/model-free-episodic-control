#!/usr/bin/env python3

import os
import random
import time

import gym

from mfec.agent import MFECAgent
from utils import Utils
from dqn.agent import DQNAgent, preprocess
from both.agent import BOTHAgent

ENVIRONMENT = "Riverraid-v0"  # More games at: https://gym.openai.com/envs/#atari
AGENT_PATH_MFEC = None#"agents/MFEC/MsPacman-v0_1609170206/agent.pkl"
AGENT_PATH_DQN = None#"agents/DQN/MsPacman-v0_1609170206/agent.pkl"

# MFEC, DQN or BOTH
ALGORITHM = 'BOTH'

# for the BOTH mode (INITIAL > FINAL)
INTRO_EPOCHS = 10.0
TRANSITION_EPOCHS = 10.0
INITIAL_MFEC_PROB = 1.0
FINAL_MFEC_PROB = 0.0
# remember to adjust the EXPLORATION_STEPS of the dqn agent

RENDER = True
RENDER_SPEED = 0.04

EPOCHS = 50
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
NO_OP_STEPS = {"DQN": 30, "MFEC": 0, "BOTH": 1}

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
            if AGENT_PATH_MFEC:
                agent = MFECAgent.load(AGENT_PATH_MFEC)
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
        elif ALGORITHM == 'DQN':
            agent = DQNAgent(env.action_space.n)
            if AGENT_PATH_DQN:
                agent.load(AGENT_PATH_DQN)
        else: # BOTH
            agent = BOTHAgent(
                INTRO_EPOCHS * FRAMES_PER_EPOCH / FRAMESKIP, TRANSITION_EPOCHS * FRAMES_PER_EPOCH / FRAMESKIP, INITIAL_MFEC_PROB, FINAL_MFEC_PROB,
                (ACTION_BUFFER_SIZE, K, DISCOUNT, EPSILON, SCALE_HEIGHT, SCALE_WIDTH, STATE_DIMENSION, range(env.action_space.n), SEED,),
                (env.action_space.n,)
                )
            # TODO loading from checkpoint
            # BTW dqn still doesn't work - the epsilon status is forgotten when loading
        
        run_algorithm(agent, agent_dir, env, utils)

    finally:
        utils.close()
        env.close()


def run_algorithm(agent, agent_dir, env, utils):
    frames_left = 0
    for _ in range(EPOCHS):
        frames_left += FRAMES_PER_EPOCH
        while frames_left > 0:
            episode_frames, episode_reward = run_episode(agent, env)
            frames_left -= episode_frames
            utils.end_episode(episode_frames, episode_reward)
        utils.end_epoch()
        agent.save(agent_dir)


def run_episode(agent, env):
    episode_frames = 0
    episode_reward = 0

    env.seed(random.randint(0, 1000000))
    observation = env.reset()

    no_op_steps = NO_OP_STEPS[ALGORITHM]
    if no_op_steps > 0:
        for _ in range(random.randint(1, no_op_steps)):
            last_observation = observation
            observation, _, _, _ = env.step(0)  # Do nothing

    if ALGORITHM == 'DQN' or ALGORITHM == 'BOTH':
        state = agent.get_initial_state(observation, last_observation)

    done = False
    while not done:

        if RENDER:
            env.render()
            time.sleep(RENDER_SPEED)

        last_observation = observation

        if ALGORITHM == 'MFEC':
            action = agent.choose_action(observation)
        elif ALGORITHM == 'DQN':
            action = agent.choose_action(state)
        else: # BOTH   
            action = agent.choose_action(observation, state)
            

        observation, reward, done, _ = env.step(action)

        if ALGORITHM == 'MFEC':
            agent.receive_reward(reward)
        else: # DQN or BOTH
            processed_observation = preprocess(observation, last_observation)
            state = agent.run(state, action, reward, done, processed_observation)

        episode_reward += reward
        episode_frames += FRAMESKIP

    if ALGORITHM == 'MFEC' or ALGORITHM == 'BOTH':
        agent.train()
    return episode_frames, episode_reward
    

if __name__ == "__main__":
    main()
