import numpy as np
from mss import mss
from PIL import Image, ImageOps
import keyboard
import time
from Restart import Restart
import tensorflow as tf
import matplotlib.pyplot as plt


class GameEnv:  # this will create an environment for the game play
    def __init__(self):

        self.Screen = {}  # setting appropriate screen sizes
        self.Screen['top'] = 350
        self.Screen['left'] = 15
        self.Screen['width'] = 1900
        self.Screen['height'] = 380
        R = Restart()
        self.img_2 = R.img
        self.sct = mss()
        self.counter = 0
        self.startTime = time.time() - 1
        self.image_list = []

        self.Default_action = 2

        self.ones = np.ones((76, 384, 4))
        self.val = [5, 1, 1, 1]
        self.set_val = []
        tam = np.zeros((76, 384, 4))
        self.set_val.append(tam)
        for i in range(1, 5):
            tam = np.zeros((76, 384, 4))
            tam[:, :, i - 1] = self.val[i - 1]
            self.set_val.append(tam)

    def start_run(self):  # just to show in the console to open the game

        print("GAME WILL START IN 3 SECOND, OPEN CHROME DINO GAME")
        # webbrowser.open('chrome://dino/', new=2)
        time.sleep(3)

    def step(self, action):  # determines the steps of agent
        actions = {}
        actions[0] = 'space'
        actions[1] = 'down'

        if action != self.Default_action:
            if self.Default_action != 2:
                keyboard.release(actions.get(self.Default_action))
            if action != 2:
                keyboard.press(actions.get(action))
        self.Default_action = action

        SS = self.sct.grab(self.Screen)
        img = np.array(SS)[:, :, 0]
        state = self.setupdata(img)
        over = self.Game_complit(img)
        reward = self.get_Reward(over)
        return state, reward, over

    def Game_reset(self):   # to get the space button moves from keyboard
        self.startTime = time.time()
        keyboard.press("space")
        time.sleep(0.5)
        keyboard.release("space")
        return self.step(0)

    def setupdata(self, img):   # initially setting the screen images as states
        img = Image.fromarray(img)
        img = img.resize((384, 76), Image.ANTIALIAS)
        if np.sum(img) > 2000000:
            img = ImageOps.invert(img)

        img = np.clip(img, 32, 172)
        img = ((img - 32) / (139))

        img = np.reshape(img, (76, 384))
        img = np.array(img)
        while len(self.image_list) < (4):
            self.image_list.append(np.reshape(img, (76, 384, 1)) * self.ones)

        bank = np.array(self.image_list)
        state = self.set_val[0]
        img1 = (np.reshape(img, (76, 384, 1)) * self.ones) * self.set_val[1]
        img2 = bank[0] * self.set_val[2]
        img3 = bank[1] * self.set_val[3]
        img4 = bank[2] * self.set_val[4]

        self.image_list.pop(0)
        self.image_list.append(np.reshape(img, (76, 384, 1)) * self.ones)

        state = np.array(img1 + img2 + img3 + img4)

        return state

    def get_Reward(self, over): # get rewards for game overs
        if over:
            return -20
        else:
            return 1

    def Game_complit(self, img):    # once the game gets over, it has to restart
        img = Image.fromarray(img)
        img = img.resize((384, 76), Image.ANTIALIAS)
        if np.sum(img) > 2000000:
            img = ImageOps.invert(img)
        img = np.clip(img, 32, 172)
        img = ((img - 32) / (139))

        img = np.reshape(img, (76, 384))

        img = np.array(img)
        img_1 = img[30:45, 191:207]
        sum = 0
        for i in range(0, len(img_1), 1):
            for j in range(len(img_1[0])):
                sum += (img_1[i][j] - self.img_2[i][j]) ** 2
                c = 0

        # print('sum is ', sum)
        # plt.imshow(img_1, interpolation='nearest')
        # plt.show()
        game_over = 90.1627243448776    # this is the value to setup for the restart
        if abs(sum - game_over) < 1:    # the game will restart automatically by checking this condition once the value is set
            print("------------------------------------------------")
            return True
        return False
