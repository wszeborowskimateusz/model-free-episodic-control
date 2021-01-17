from mfec.agent import MFECAgent
from dqn.agent import DQNAgent
import random
import os


class BOTHAgent():
    def __init__(self, introSteps, transitionSteps, initialMfecProb, finalMfecProb, mfecAgentParams, dqnAgentParams):
        self.introStepsLeft = introSteps
        self.mfecProb = initialMfecProb
        self.targetMfecProb = finalMfecProb
        self.probStep = (initialMfecProb - finalMfecProb) / transitionSteps
        
        self.mfecAgent = MFECAgent(*mfecAgentParams)
        self.dqnAgent = DQNAgent(*dqnAgentParams)

    def get_initial_state(self, observation, last_observation):
        return self.dqnAgent.get_initial_state(observation, last_observation)

    def choose_action(self, observation, state):

        mfecAction = self.mfecAgent.choose_action(observation)
        dqnAction = self.dqnAgent.choose_action(state)
        if random.random() <= self.mfecProb:
            action = mfecAction
        else:
            action = dqnAction
        
        if self.introStepsLeft > 0:
            self.introStepsLeft -= 1
        elif self.mfecProb > self.targetMfecProb:
            self.mfecProb -= self.probStep
        
        return action

    def run(self, state, action, reward, terminal, processed_observation):
        self.mfecAgent.receive_reward(reward)
        return self.dqnAgent.run(state, action, reward, terminal, processed_observation)

    def train(self):
        self.mfecAgent.train()
        print(f"Choosing MFEC action in {100*self.mfecProb:.2f}% of cases.")

    def save(self, results_dir):

        mfecDir = f"{results_dir}\MFEC"
        dqnDir = f"{results_dir}\DQN"

        if not os.path.exists(mfecDir):
            os.makedirs(mfecDir)
        if not os.path.exists(dqnDir):
            os.makedirs(dqnDir)
        self.mfecAgent.save(mfecDir)
        self.dqnAgent.save(dqnDir)