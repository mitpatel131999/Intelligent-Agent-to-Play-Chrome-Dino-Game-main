from DianosorAgent import DianosorAgent
from Train import Train
from Test import Test
from Restart import Restart
from GameEnv import GameEnv

R = Restart()
img = R.img

while True:
    print("  ")
    print(" 1 .   Train Model ")  # getting user's choice
    print(" 2 .   Test Model ")
    print(" 3 .   Quit        ")
    print(" Entet the choice:=")
    x = int(input())  # based on the input given by user, the test or train functions will be called
    if x == 1:  # If the user given train the model
        print("how many number of games ? -- ")
        N = int(input())
        A = DianosorAgent
        B = GameEnv()
        T = Train(A, B, N)
        T.start()
        T.plot_performance()
    if x == 2:  # If the user given test the model
        print("how many number of games ? -- ")
        N = int(input())
        A = DianosorAgent
        B = GameEnv()
        T = Test(A, B, N)
        T.start()
        T.plot_performance()
    if x == 3:  # for quitting the program
        break
    if x != 1 and x != 2 and x != 3:  # if the user entered the invalid option
        print("Enter Right Choice")
