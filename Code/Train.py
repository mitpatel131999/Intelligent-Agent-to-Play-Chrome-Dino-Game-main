import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from DianosorAgent import DianosorAgent
from GameEnv import GameEnv
import  tensorflow as tf


class Train:    # training phase of the agent
    def __init__(self,A,B,N=2000):
        self.c=0
        self.N=N
        self.agent=A
        self.env=B
        self.plotX=[]

    def start(self):    # training will start here with storing all its experiences of agent
        self.plotX = []

        self.agent = DianosorAgent()
        self.env = GameEnv()
        self.env.start_run()
        for i in tqdm(range(self.N)):
                state, reward, done = self.env.Game_reset()
                epReward = 0
                done = False

                stepCounter = 0
                while not done:

                    action = self.agent.action(state)
                    nextState, reward, done = self.env.step(action)

                    for p in range( max(1,stepCounter//50)):
                            self.agent.remember(state, nextState, action, reward, done, stepCounter)
                    if done == True: 
                        for p in range(10):
                            self.agent.remember(state, nextState, action, reward, done, stepCounter)
                        break
                    state = nextState
                    stepCounter += 1
                    epReward += reward

                self.plotX.append(epReward)
                self.agent.learn()
                self.agent.diano_model.save_weights("DinoTrain.h5")

    def plot_performance(self):
        plt.plot(range(len(self.plotX)),self.plotX) 
        plt.show()

