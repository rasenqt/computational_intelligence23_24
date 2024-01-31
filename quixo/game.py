from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import torch
from torch import nn
import random
import numpy as np
import torch
import torch.nn.init as init

# Rules on PDF

class QuixoNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5*5, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 44),
        )
        for layer in self.linear_relu_stack:
                    if isinstance(layer, nn.Linear):
                        init.xavier_uniform_(layer.weight)
                        
    def forward(self, x):
        logits = self.linear_relu_stack(x)
        return logits

class Move(Enum):
    '''
    Selects where you want to place the taken piece. The rest of the pieces are shifted
    '''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    

from abc import abstractmethod


class Player(ABC):
    def __init__(self) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass
    
    @abstractmethod
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        The game accepts coordinates of the type (X, Y). X goes from left to right, while Y goes from top to bottom, as in 2D graphics.
        Thus, the coordinates that this method returns shall be in the (X, Y) format.

        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
    pass


class RandomPlayer(Player):
    def __init__(self) -> None:
        super().__init__()

    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
       ok=False
       #try_game = deepcopy(game)

       while not ok:
              try_game = deepcopy(game)
              from_pos = (random.randint(0, 4), random.randint(0, 4))
              move = random.choice([Move.TOP, Move.BOTTOM, Move.LEFT, Move.RIGHT])
              ok = try_game._Game__move(from_pos, move, game.current_player_idx)   
            
       return (from_pos, move)


class MyPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.GeneratorNet = QuixoNet().to(self.device)
        self.TargetNet = QuixoNet().to(self.device)
        self.criterion = nn.SmoothL1Loss().to(self.device)
        self.optimizer = torch.optim.Adam(self.GeneratorNet.parameters(), lr=0.01)

        self.last_action_value = 0.0
        self.last_action_number=0
        self.step=1.0
        
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        nn_input = game.get_flat_board()
        nn_input = torch.tensor(nn_input, dtype=torch.float32).to(self.device)
        out = self.GeneratorNet(nn_input) # ritorna 44 uscite, da qui ricavare il numero dell'azione e la direzione della migliori
        out = out.cpu().detach().numpy()
        index = np.argsort(out)
        pos,move=0,0
        from_pos=(0,0)
        sorted_index = index[::-1]
      
        for _index in sorted_index:
            try_game = deepcopy(game)
            pos, move = translate_number_to_position_direction(_index+1) #perchè index va da 0 a 43 noi abbiam mappato da 1 a 44
            from_pos = translate_number_to_position(pos) #from_pos =Tuple[int,int]direction
            self.last_action_value = out[_index]
            self.last_action_number=_index

            ok = try_game._Game__move(from_pos, move, game.current_player_idx)
            
            if ok:
                break    

        return  (from_pos, move)
    
    
    def myplayer_zero_grad(self):
          self.optimizer.zero_grad()
          
    def downstream(self,game: 'Game',net: QuixoNet):
         tensor=torch.tensor(game.get_flat_board(), dtype=torch.float32).to(self.device)
         output=net(tensor)
         return output
         
    def myplayer_loss_and_update_params(self, GeneratorNet_outputs, TargetNet_targets):

        loss = self.criterion(GeneratorNet_outputs, TargetNet_targets)
        loss.backward()
        self.optimizer.step()
        return loss
        
    def copy_params_TargetNet(self):
        self.TargetNet.load_state_dict(self.GeneratorNet.state_dict())
        
    def compute_target(self, game: 'Game'):
        board = game.get_flat_board()
        nn_input = torch.tensor(board, dtype=torch.float32).to(self.device)
        out = self.TargetNet(nn_input) # ritorna 44 uscite, da qui ricavare il numero dell'azione e la direzione della migliori
        out = out.cpu().detach().numpy()
        index = np.argsort(out)
        pos,move=0,0
        from_pos=(0,0)
        sorted_index = index[::-1]
        action_val = 0
       
        for _index in sorted_index:
            pos, move = translate_number_to_position_direction(_index+1) ##perchè index va da 0 a 43 noi abbiam mappato da 1 a 44
            from_pos=translate_number_to_position(pos)#from_pos =Tuple[int,int]direction
            action_val = out[_index]
            ok = game._Game__move(from_pos, move, game.current_player_idx)   
            if ok:
                break 
             
        return  action_val



class TrainedPlayer(Player):
    def __init__(self) -> None:
        super().__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.GeneratorNet = QuixoNet().to(self.device)
       
        
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        nn_input = game.get_flat_board()
        nn_input = torch.tensor(nn_input, dtype=torch.float32).to(self.device)
        out = self.GeneratorNet(nn_input) # ritorna 44 uscite, da qui ricavare il numero dell'azione e la direzione della migliori
        out = out.cpu().detach().numpy()
        index = np.argsort(out)
        pos,move=0,0
        from_pos=(0,0)
        sorted_index = index[::-1]
       
        for _index in sorted_index:
            try_game = deepcopy(game)
            pos, move = translate_number_to_position_direction(_index+1) #perchè index va da 0 a 43 noi abbiam mappato da 1 a 44
            from_pos = translate_number_to_position(pos) #from_pos =Tuple[int,int]direction
            self.last_action_value = out[_index]
            self.last_action_number=_index

            ok = try_game._Game__move(from_pos, move, game.current_player_idx)
            
            if ok:
                break    

        return  (from_pos, move)
    
class TrainedPlayer_Complete(Player):
    def __init__(self, isfirst, path_first, path_second) -> None:
        super().__init__()
        self.device = ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.GeneratorNet_first = QuixoNet().to(self.device)
        self.GeneratorNet_second = QuixoNet().to(self.device)
        self.GeneratorNet_first.load_state_dict(torch.load(path_first))
        self.GeneratorNet_second.load_state_dict(torch.load(path_second))
        self.isfirst = isfirst
       
    def make_move(self, game: 'Game') -> tuple[tuple[int, int], Move]:
        nn_input = game.get_flat_board()
        nn_input = torch.tensor(nn_input, dtype=torch.float32).to(self.device)
        out = self.GeneratorNet_first(nn_input) if self.isfirst else self.GeneratorNet_second(nn_input) # ritorna 44 uscite, da qui ricavare il numero dell'azione e la direzione della migliori
        out = out.cpu().detach().numpy()
        index = np.argsort(out)
        pos,move=0,0
        from_pos=(0,0)
        sorted_index = index[::-1]

        for _index in sorted_index:
            try_game = deepcopy(game)
            pos, move = translate_number_to_position_direction(_index+1) # perchè index va da 0 a 43 noi abbiam mappato da 1 a 44
            from_pos = translate_number_to_position(pos) # from_pos =Tuple[int,int]direction
            ok = try_game._Game__move(from_pos, move, game.current_player_idx)
            
            if ok:
                break    

        return  (from_pos, move)
    

    def change_turn(self, turn: bool):
        self.isfirst = turn

    
def translate_number_to_position_direction(number)->tuple[int,Move]:
        #CASELLA 1 TOP LEFT CORNER
        if number == 1:
            return (1, Move.BOTTOM)
        elif number == 2:
               return (1, Move.RIGHT)
        #TOP----------------------------
        #CASELLA 2
        elif number == 3:
               return (2, Move.BOTTOM)
        elif number == 4:
               return (2, Move.RIGHT)
        elif number == 5:
               return (2, Move.LEFT)
        
        #CASELLA 3
        elif number == 6:
               return (3, Move.BOTTOM)
        elif number == 7:
               return (3, Move.RIGHT)
        elif number == 8:
               return (3, Move.LEFT)
        #CASELLA 4
        elif number == 9:
               return (4, Move.BOTTOM)
        elif number == 10:
               return (4, Move.RIGHT)
        elif number == 11:
               return (4, Move.LEFT)
        #------------------------------
        #CASELLA 5 -TOP RIGHT CORNER
        elif number == 12:
               return (5, Move.BOTTOM)
        elif number == 13:
               return (5, Move.LEFT)
        #RIGHT-------------
        #CASELLA 6
        elif number == 14:
               return (6, Move.BOTTOM)
        elif number == 15:
               return (6, Move.TOP)
        elif number == 16:
               return (6, Move.LEFT)
        
        #CASELLA 7
        elif number == 17:
               return (7, Move.BOTTOM)
        elif number == 18:
               return (7, Move.TOP)
        elif number == 19:
               return (7, Move.LEFT)
        

        #CASELLA 8
        elif number == 20:
               return (8, Move.BOTTOM)
        elif number == 21:
               return (8, Move.TOP)
        elif number == 22:
               return (8, Move.LEFT)
        #-----------------------
        #CASELLA 9 BOTTOM RIGHT
        elif number == 23:
               return (9, Move.TOP)
        elif number == 24:
               return (9, Move.LEFT)
        #-BOTTOM-------
        #CASELLA 10
        elif number == 25:
               return (10, Move.RIGHT)
        elif number == 26:
               return (10, Move.TOP)
        elif number == 27:
               return (10, Move.LEFT)
         #CASELLA 11
        elif number == 28:
               return (11, Move.RIGHT)
        elif number == 29:
               return (11, Move.TOP)
        elif number == 30:
               return (11, Move.LEFT)
         #CASELLA 12
        elif number == 31:
               return (12, Move.RIGHT)
        elif number == 32:
               return (12, Move.TOP)
        elif number == 33:
               return (12, Move.LEFT)
        ##CASELLA 13 BOTTOM LEFT
        elif number == 34:
               return (13, Move.TOP)
        elif number == 35:
               return (13, Move.RIGHT)
        #LEFT----------------
        #CASELLA 14
        elif number == 36:
               return (14, Move.RIGHT)
        elif number == 37:
               return (14, Move.TOP)
        elif number == 38:
               return (14, Move.BOTTOM)
         #CASELLA 15
        elif number == 39:
               return (15, Move.RIGHT)
        elif number == 40:
               return (15, Move.TOP)
        elif number == 41:
               return (15, Move.BOTTOM)
         #CASELLA 16
        elif number == 42:
               return (16, Move.RIGHT)
        elif number == 43:
               return (16, Move.TOP)
        elif number == 44:
               return (16, Move.BOTTOM)

def translate_number_to_position(number)->tuple[int, int]:
       '''Translate Position (1-16) into row,col'''
       if 1 <= number <= 16:
        if number <= 5:
            return (0, number - 1)
        elif number <= 9:
            return (number - 5, 4)
        elif number <= 13:
            return (4, 4 - (number - 9))
        elif number <= 16:
            return (4 - (number - 13), 0)
       else:
         return None
       
def translate_position_to_number(position: tuple[int, int]) -> int:
    '''Translate row, col into Position (1-16)'''
    row, col = position
    if 0 <= row <= 4 and 0 <= col <= 4:
        if row == 0:
            return col + 1
        elif col == 4:
            return 5 + row
        elif row == 4:
            return 9 + (4 - col)
        elif col == 0:
            return 13 + (4 - row)
    return None
       
       

class Game(object):
    def __init__(self) -> None:
        self._board = np.ones((5, 5), dtype=np.uint8) * -1
        self.current_player_idx = 0

    def get_board(self) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(self._board)
    
    def get_flat_board(self) : # DA SPOSTARE
        nn_input = deepcopy(self._board.flatten())
        return nn_input

    def get_current_player(self) -> int: 
        '''
        Returns the current player
        '''
        return deepcopy(self.current_player_idx)

    def print(self):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        print(self._board)

    def check_winner(self) -> int:
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        for x in range(self._board.shape[0]):
            # if a player has completed an entire row
            if self._board[x, 0] != -1 and all(self._board[x, :] == self._board[x, 0]):
                # return the relative id
                return self._board[x, 0]
        # for each column
        for y in range(self._board.shape[1]):
            # if a player has completed an entire column
            if self._board[0, y] != -1 and all(self._board[:, y] == self._board[0, y]):
                # return the relative id
                return self._board[0, y]
        # if a player has completed the principal diagonal
        if self._board[0, 0] != -1 and all(
            [self._board[x, x]
                for x in range(self._board.shape[0])] == self._board[0, 0]
        ):
            # return the relative id
            return self._board[0, 0]
        # if a player has completed the secondary diagonal
        if self._board[0, -1] != -1 and all(
            [self._board[x, -(x + 1)]
             for x in range(self._board.shape[0])] == self._board[0, -1]
        ):
            # return the relative id
            return self._board[0, -1]
        return -1
    

    def play(self, player1: Player, player2: Player):
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            self.current_player_idx += 1
            self.current_player_idx %= len(players)
            ok = False
            
            while not ok:
                from_pos, slide = players[self.current_player_idx].make_move(
                        self)
                ok = self._Game__move(from_pos, slide, self.current_player_idx)
                
            winner = self.check_winner()
        return winner

    
    def __move(self, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(self._board[(from_pos[0], from_pos[1])])
        acceptable = self.__take((from_pos[0], from_pos[1]), player_id)
        if acceptable:
            acceptable = self.__slide((from_pos[0], from_pos[1]), slide)
            if not acceptable:
                self._board[(from_pos[0], from_pos[1])] = deepcopy(prev_value)
        return acceptable
    

    def __take(self, from_pos: tuple[int, int], player_id: int) -> bool:
        '''Take piece'''
        # acceptable only if in border
        acceptable: bool = (
            # check if it is in the first row
            (from_pos[0] == 0 and from_pos[1] < 5)
            # check if it is in the last row
            or (from_pos[0] == 4 and from_pos[1] < 5)
            # check if it is in the first column
            or (from_pos[1] == 0 and from_pos[0] < 5)
            # check if it is in the last column
            or (from_pos[1] == 4 and from_pos[0] < 5)
            # and check if the piece can be moved by the current player
        ) and (self._board[from_pos] < 0 or self._board[from_pos] == player_id)
        if acceptable:
            self._board[from_pos] = player_id
        return acceptable
            

    def __slide(self, from_pos: tuple[int, int], slide: Move) -> bool:
            '''Slide the other pieces'''
            # define the corners
            SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
            # if the piece position is not in a corner
            if from_pos not in SIDES:
                # if it is at the TOP, it can be moved down, left or right
                acceptable_top: bool = from_pos[0] == 0 and (
                    slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
                )
                # if it is at the BOTTOM, it can be moved up, left or right
                acceptable_bottom: bool = from_pos[0] == 4 and (
                    slide == Move.TOP or slide == Move.LEFT or slide == Move.RIGHT
                )
                # if it is on the LEFT, it can be moved up, down or right
                acceptable_left: bool = from_pos[1] == 0 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
                )
                # if it is on the RIGHT, it can be moved up, down or left
                acceptable_right: bool = from_pos[1] == 4 and (
                    slide == Move.BOTTOM or slide == Move.TOP or slide == Move.LEFT
                )
            # if the piece position is in a corner
            else:
                # if it is in the upper left corner, it can be moved to the right and down
                acceptable_top: bool = from_pos == (0, 0) and (
                    slide == Move.BOTTOM or slide == Move.RIGHT)
                # if it is in the lower left corner, it can be moved to the right and up
                acceptable_left: bool = from_pos == (4, 0) and (
                    slide == Move.TOP or slide == Move.RIGHT)
                # if it is in the upper right corner, it can be moved to the left and down
                acceptable_right: bool = from_pos == (0, 4) and (
                    slide == Move.BOTTOM or slide == Move.LEFT)
                # if it is in the lower right corner, it can be moved to the left and up
                acceptable_bottom: bool = from_pos == (4, 4) and (
                    slide == Move.TOP or slide == Move.LEFT)
            # check if the move is acceptable
            acceptable: bool = acceptable_top or acceptable_bottom or acceptable_left or acceptable_right
            # if it is
            if acceptable:
                # take the piece
                piece = self._board[from_pos]
                # if the player wants to slide it to the left
                if slide == Move.LEFT:
                    # for each column starting from the column of the piece and moving to the left
                    for i in range(from_pos[1], 0, -1):
                        # copy the value contained in the same row and the previous column
                        self._board[(from_pos[0], i)] = self._board[(
                            from_pos[0], i - 1)]
                    # move the piece to the left
                    self._board[(from_pos[0], 0)] = piece
                # if the player wants to slide it to the right
                elif slide == Move.RIGHT:
                    # for each column starting from the column of the piece and moving to the right
                    for i in range(from_pos[1], self._board.shape[1] - 1, 1):
                        # copy the value contained in the same row and the following column
                        self._board[(from_pos[0], i)] = self._board[(
                            from_pos[0], i + 1)]
                    # move the piece to the right
                    self._board[(from_pos[0], self._board.shape[1] - 1)] = piece
                # if the player wants to slide it upward
                elif slide == Move.TOP:
                    # for each row starting from the row of the piece and going upward
                    for i in range(from_pos[0], 0, -1):
                        # copy the value contained in the same column and the previous row
                        self._board[(i, from_pos[1])] = self._board[(
                            i - 1, from_pos[1])]
                    # move the piece .
                        
                        
                    self._board[(0, from_pos[1])] = piece
                # if the player wants to slide it downward
                elif slide == Move.BOTTOM:
                    # for each row starting from the row of the piece and going downward
                    for i in range(from_pos[0], self._board.shape[0] - 1, 1):
                        # copy the value contained in the same column and the following row
                        self._board[(i, from_pos[1])] = self._board[(
                            i + 1, from_pos[1])]
                    # move the piece down
                    self._board[(self._board.shape[0] - 1, from_pos[1])] = piece
            return acceptable



    def  compute_reward(self) -> int :
        reward=0.1

        if self.check_winner()==0:
            reward = 1
        if self.check_winner()==1:
            reward =-1
        
        return reward
        
    

       
