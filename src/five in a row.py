import numpy as np
import cv2
import math

BLACK_WIN_SIGN = np.array([1.0,1.0,1.0,1.0,1.0])
WHITE_WIN_SIGN = np.array([2.0,2.0,2.0,2.0,2.0])
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


##def return_state():
##    state = np.zeros(len(board),len(board),3)
##    for x in range(len(board)):
##        for y in range(len(board)):
##            if board[x][y] == 1:
##                state[x][y][0] = 1
##            if board[x][y] == 2:
##                state[x][y][1] = 1
##            state[x][y][2] = 0
    
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
                state[x][y][2] = 0
        return state, -1, False
    board[x][y] = 1
    for x1 in range(len(board)):
        for y1 in range(len(board)):
            try:
##                print('checked')
                if all([board[x1][y1], board[x1+1][y1], board[x1+2][y1], board[x1+3][y1], board[x1+4][y1]] == BLACK_WIN_SIGN) or all([board[x1][y1], board[x1+1][y1+1], board[x1+2][y1+2], board[x1+3][y1+3], board[x1+4][y1+4]] == BLACK_WIN_SIGN) or all([board[x1][y1], board[x1][y1+1], board[x1][y1+2], board[x1][y1+3], board[x1][y1+4]] == BLACK_WIN_SIGN):
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
##                print('passed')
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
                state[x][y][2] = 1
        return state, -1, False
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

done = False
game = reset()
current_state = np.zeros((19,19,3))
while not done:
    draw_img(game)
    action1 = int(input("black's turn:"))
    new_state1, reward1, done = ai_step1(action1,game)
    draw_img(game)
    if reward1 != 1:
        action2 = int(input("white's turn:"))
        new_state2,reward2,done = ai_step2(action2,game)
        draw_img(game)
        if reward2 == 1:
            reward1 = -1
            print('white won!')
    elif reward1 == 1:
        reward2 = -1
        print('black won!')
    print(reward1,reward2)
    
    
