import time
import matplotlib.pyplot as plt
from tqdm import tqdm
import  tensorflow as tf
from DianosorAgent import DianosorAgent
from GameEnv import GameEnv


class Test: # this is the testing phase of agent
    def __init__(self,A,B,N):
        self.c=0
        self.agent=A
        self.env=B
        self.plotX=[]
        self.N=N

    def start(self):    # testing will be done here without storing its experiences into DinoTrain.h5 file

        for i in range (self.N):
            self.agent = DianosorAgent()
            self.env = GameEnv()
            self.env.start_run()
            done = False
            reward=0
            start=time.time()
            state, rewar, done = self.env.Game_reset()
            while not done:
                    start= time.time()
                    action = self.agent.action(state)
                    nextState, score, done = self.env.step(action)
                    reward+=score
                    if done == True:
                        print("Game over",i)
                        break

                    state = nextState
            end = time.time()
            print('score   -> ',end-start)
            self.plotX.append(end-start)

    def plot_performance(self):

        plt.plot(range(len(self.plotX)), self.plotX)
        plt.show()


