import numpy as np
import cv2
from collections import deque
import math
import random
from keras.models import Sequential, load_model
from keras.layers import *
from keras.optimizers import RMSprop

BLACK_WIN_SIGN = np.array([1.0,1.0,1.0,1.0,1.0])
WHITE_WIN_SIGN = np.array([2.0,2.0,2.0,2.0,2.0])
MODEL = None


def reset():
    board_size = 15
    game = np.zeros((board_size,board_size))
    return game

def draw_img(board):
    gap = 15
    real_img = np.zeros((500,500,3))
    for x in range(int(len(real_img))):
        for y in range(int(len(real_img))):
            real_img[x][y] = (0, 255, 0)
                       
    for x1 in range(len(board)):
        for y1 in range(len(real_img)):
            real_img[int(len(real_img)/len(board)*x1)+gap][y1] = (0,0,0)
            real_img[y1][int(len(real_img)/len(board)*x1)+gap] = (0,0,0)

    for x2 in range(len(board)):
        for y2 in range(len(board)):
            if board[x2][y2] == 1:
                cv2.circle(real_img, (int(((len(real_img)/len(board))*x2)+gap),int(((len(real_img)/len(board))*y2))+gap),
                           15, (0,0,0),-1)
            if board[x2][y2] == 2:
                cv2.circle(real_img, (int(((len(real_img)/len(board))*x2)+gap),int(((len(real_img)/len(board))*y2))+gap),
                           15, (255, 255, 255),-1)
    cv2.imshow('game',real_img)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
            
    return real_img
    
def ai_step1(input_n,board): 
    x = math.floor(input_n/len(board))
    y = input_n % len(board)

    if board[x][y] != 0:
        state = np.zeros((len(board),len(board),3))
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == 1:
                    state[x][y][0] = 1
                if board[x][y] == 2:
                    state[x][y][1] = 1
                state[x][y][2] = 1
        return state, -1, True
    board[x][y] = 1
    for x1 in range(len(board)):
        for y1 in range(len(board)):
            try:
                if all([board[x1][y1], board[x1+1][y1], board[x1+2][y1], board[x1+3][y1], board[x1+4][y1]] == BLACK_WIN_SIGN) or all([board[x1][y1], board[x1+1][y1+1], board[x1+2][y1+2], board[x1+3][y1+3], board[x1+4][y1+4]] == BLACK_WIN_SIGN) or all([board[x1][y1], board[x1][y1+1], board[x1][y1+2], board[x1][y1+3], board[x1][y1+4]] == BLACK_WIN_SIGN):
                    state = np.zeros((len(board),len(board),3))
                    for x in range(len(board)):
                        for y in range(len(board)):
                            if board[x][y] == 1:
                                state[x][y][0] = 1
                            if board[x][y] == 2:
                                state[x][y][1] = 1
                            state[x][y][2] = 1
                    return state, 1, True
            except:
                pass                  
    else:
        state = np.zeros((len(board),len(board),3))
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == 1:
                    state[x][y][0] = 1
                if board[x][y] == 2:
                    state[x][y][1] = 1
                state[x][y][2] = 1
        return state, 0, False

def ai_step2(input_n,board): 
    x = math.floor(input_n/len(board))
    y = input_n % len(board)

    if board[x][y] != 0:
        state = np.zeros((len(board),len(board),3))
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == 1:
                    state[x][y][0] = 1
                if board[x][y] == 2:
                    state[x][y][1] = 1
                state[x][y][2] = 0
        return state, -1, True
    board[x][y] = 2
    for x1 in range(len(board)):
        for y1 in range(len(board)):
            try:
                if all([board[x1][y1], board[x1+1][y1], board[x1+2][y1], board[x1+3][y1], board[x1+4][y1]] == WHITE_WIN_SIGN) or all([board[x1][y1], board[x1+1][y1+1], board[x1+2][y1+2], board[x1+3][y1+3], board[x1+4][y1+4]] == WHITE_WIN_SIGN) or all([board[x1][y1], board[x1][y1+1], board[x1][y1+2], board[x1][y1+3], board[x1][y1+4]] == WHITE_WIN_SIGN):
                    state = np.zeros((len(board),len(board),3))
                    for x in range(len(board)):
                        for y in range(len(board)):
                            if board[x][y] == 1:
                                state[x][y][0] = 1
                            if board[x][y] == 2:
                                state[x][y][1] = 1
                            state[x][y][2] = 0
                    return state, 1, True
            except:
                pass                  
    else:
        state = np.zeros((len(board),len(board),3))
        for x in range(len(board)):
            for y in range(len(board)):
                if board[x][y] == 1:
                    state[x][y][0] = 1
                if board[x][y] == 2:
                    state[x][y][1] = 1
                state[x][y][2] = 0
        return state, 0, False
    
def train(model,target_model,transition):
    MIN_REPLAY_SIZE = 100
    MINIBATCH_SIZE = 16
    if transition < MIN_REPLAY_SIZE:
        return
    minibatch = random.sample(transition,MINIBATCH_SIZE)
    X = []
    y = []
    
    
def create_model():

    if MODEL is not None:
        model = load_model(MODEL)
    else:
        model = Sequential()

        model.add(Conv2D(256,(3,3),input_shape=(15,15,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Conv2D(256,(3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Conv2D(256,(3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Conv2D(256,(3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Conv2D(256,(3,3)))
        model.add(MaxPooling2D(pool_size=(2,2)))
        model.add(Activation('relu'))

        model.add(Flatten())
        model.add(Dense(512,activation='relu'))
        model.add(Dense(225))

    return model

def main():    
    done = False
    game = reset()
    state0 = np.zeros((15,15,3))
    train_set = deque(maxlen=100_000_000)
    EPISODES = 50000
    net = create_model()
    tar_net = create_model()
    tar_net.set_weights(net.get_weights())
    UPDATE_FREQ = 1000
    counter = 0
    for episode in range(EPISODES):
        while not done:
            counter += 1
            draw_img(game)
            action1 = np.argmax(net.predict(state0.reshape(1,15,15,3))[0])
            new_state1, reward1, done1 = ai_step1(action1,game)
            draw_img(game)
            if reward1 != 1:
                action2 = np.argmax(net.predict(new_state1.reshape(1,15,15,3))[0])
                new_state2,reward2,done2 = ai_step2(action2,game)
                draw_img(game)
                if reward2 == 1:
                    reward1 = -1
                    print('white won!')
            elif reward1 == 1:
                reward2 = -1
                print('black won!')
            if done1 or done2:
                done = True
            
            train_set.append([state0,reward1,done1,action1,new_state1])
            try:
                train_set.append([new_state1,reward2,done2,action2,new_state2])
            except:
                pass
            train(net,tar_net,train_set)
            if counter >= 1000:
                counter = 0
                tar_net.set_weights(net.get_weights())
            state0 = new_state2
        
    
    
