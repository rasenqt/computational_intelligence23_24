from abc import ABC, abstractmethod
from copy import deepcopy
from enum import Enum
import numpy as np

# Rules on PDF


class Move(Enum):
    '''
    Selects where you want to place the taken piece. The rest of the pieces are shifted
    '''
    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3


class Player(ABC):
    def __init__(tmp_game) -> None:
        '''You can change this for your player if you need to handle state/have memory'''
        pass

    @abstractmethod
    def make_move(tmp_game, game: 'Game') -> tuple[tuple[int, int], Move]:
        '''
        The game accepts coordinates of the type (X, Y). X goes from left to right, while Y goes from top to bottom, as in 2D graphics.
        Thus, the coordinates that this method returns shall be in the (X, Y) format.

        game: the Quixo game. You can use it to override the current game with yours, but everything is evaluated by the main game
        return values: this method shall return a tuple of X,Y positions and a move among TOP, BOTTOM, LEFT and RIGHT
        '''
        pass


class Game(object):
    def __init__(tmp_game) -> None:
        tmp_game._board = np.ones((5, 5), dtype=np.uint8) * -1
        tmp_game.current_player_idx = 0

    def get_board(tmp_game) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(tmp_game._board)
    
    def get_flat_board(tmp_game) -> np.ndarray:
        '''
        Returns the board
        '''
        return deepcopy(tmp_game._board.flatten())

    def get_current_player(tmp_game) -> int:
        '''
        Returns the current player
        '''
        return deepcopy(tmp_game.current_player_idx)

    def print(tmp_game):
        '''Prints the board. -1 are neutral pieces, 0 are pieces of player 0, 1 pieces of player 1'''
        print(tmp_game._board)

    def check_winner(tmp_game) -> int:  #return 1 for player 1 or 0 for player 0
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        # for each row
        for x in range(tmp_game._board.shape[0]):
            # if a player has completed an entire row
            if tmp_game._board[x, 0] != -1 and all(tmp_game._board[x, :] == tmp_game._board[x, 0]):
                # return the relative id
                return tmp_game._board[x, 0]
        # for each column
        for y in range(tmp_game._board.shape[1]):
            # if a player has completed an entire column
            if tmp_game._board[0, y] != -1 and all(tmp_game._board[:, y] == tmp_game._board[0, y]):
                # return the relative id
                return tmp_game._board[0, y]
        # if a player has completed the principal diagonal
        if tmp_game._board[0, 0] != -1 and all(
            [tmp_game._board[x, x]
                for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, 0]
        ):
            # return the relative id
            return tmp_game._board[0, 0]
        # if a player has completed the secondary diagonal
        if tmp_game._board[0, -1] != -1 and all(
            [tmp_game._board[x, -(x + 1)]
             for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, -1]
        ):
            # return the relative id
            return tmp_game._board[0, -1]
        return -1
    


    def check_is_winning(self,from_pos,slide) -> int:  #return 1 for player 1 or 0 for player 0
        '''Check the winner. Returns the player ID of the winner if any, otherwise returns -1'''
        tmp_game=deepcopy(self)
        accetable=tmp_game.__slide(from_pos,slide);
        #def __move(tmp_game, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        # for each row
        for x in range(tmp_game._board.shape[0]):
            # if a player has completed an entire row
            if tmp_game._board[x, 0] != -1 and all(tmp_game._board[x, :] == tmp_game._board[x, 0]):
                # return the relative id
                return tmp_game._board[x, 0]
        # for each column
        for y in range(tmp_game._board.shape[1]):
            # if a player has completed an entire column
            if tmp_game._board[0, y] != -1 and all(tmp_game._board[:, y] == tmp_game._board[0, y]):
                # return the relative id
                return tmp_game._board[0, y]
        # if a player has completed the principal diagonal
        if tmp_game._board[0, 0] != -1 and all(
            [tmp_game._board[x, x]
                for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, 0]
        ):
            # return the relative id
            return tmp_game._board[0, 0]
        # if a player has completed the secondary diagonal
        if tmp_game._board[0, -1] != -1 and all(
            [tmp_game._board[x, -(x + 1)]
             for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, -1]
        ):
            # return the relative id
            return tmp_game._board[0, -1]
        return -1
    
    def check_2_pattern(self) -> int:  #return 1 for player 1 or 0 for player 0
        
            '''Check if the move get a in a state with at least 2 simbols in a row, otherwise returns -1'''
            n=2
           
         
        # for each row
            for x in range(self._board.shape[0]):
                # if a player has completed an entire row
                if self._board[x, 0] != -1 and sum(self._board[x, :] == self._board[x, 0]) >= n:
                    # self the relative id
                    return self._board[x, 0]
            # for each column
            for y in range(self._board.shape[1]):
                # if a player has completed an entire column
                if tmp_game._board[0, y] != -1 and sum(self._board[x, :] == self._board[x, 0]) >= n:
                    # return the relative id
                    return tmp_game._board[0, y]
            # if a player has completed the principal diagonal
            if self._board[0, 0] != -1 and sum(
                [self._board[x, x]
                    for x in range(self._board.shape[0])] == self._board[0, 0]
            )>=n:
                # return the relative id
                return self._board[0, 0]
            # if a player has completed the secondary diagonal
            if self._board[0, -1] != -1 and sum(
                [self._board[x, -(x + 1)]
                for x in range(self._board.shape[0])] == self._board[0, -1]
            )>=n:
                # return the relative id
                return self._board[0, -1]
            return -1
    
    def check_3_pattern(self,from_pos,slide) -> int:  #return 1 for player 1 or 0 for player 0
        
            '''Check if the move get a in a state with at least 3 simbols in a row, otherwise returns -1'''
            n=3
            tmp_game=deepcopy(self)
            accetable=tmp_game.__slide(from_pos,slide);
        # for each row
            for x in range(tmp_game._board.shape[0]):
                # if a player has completed an entire row
                if tmp_game._board[x, 0] != -1 and sum(tmp_game._board[x, :] == tmp_game._board[x, 0]) >= n:
                    # return the relative id
                    return tmp_game._board[x, 0]
            # for each column
            for y in range(tmp_game._board.shape[1]):
                # if a player has completed an entire column
                if tmp_game._board[0, y] != -1 and sum(tmp_game._board[x, :] == tmp_game._board[x, 0]) >= n:
                    # return the relative id
                    return tmp_game._board[0, y]
            # if a player has completed the principal diagonal
            if tmp_game._board[0, 0] != -1 and sum(
                [tmp_game._board[x, x]
                    for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, 0]
            )>=n:
                # return the relative id
                return tmp_game._board[0, 0]
            # if a player has completed the secondary diagonal
            if tmp_game._board[0, -1] != -1 and sum(
                [tmp_game._board[x, -(x + 1)]
                for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, -1]
            )>=n:
                # return the relative id
                return tmp_game._board[0, -1]
            return -1
    
    def check_4_pattern(self,from_pos,slide) -> int:  #return 1 for player 1 or 0 for player 0
        
            '''Check if the move get a in a state with at least 4 simbols in a row, otherwise returns -1'''
            n=4
            tmp_game=deepcopy(self)
            accetable=tmp_game.__slide(from_pos,slide);
        # for each row
            for x in range(tmp_game._board.shape[0]):
                # if a player has completed an entire row
                if tmp_game._board[x, 0] != -1 and sum(tmp_game._board[x, :] == tmp_game._board[x, 0]) >= n:
                    # return the relative id
                    return tmp_game._board[x, 0]
            # for each column
            for y in range(tmp_game._board.shape[1]):
                # if a player has completed an entire column
                if tmp_game._board[0, y] != -1 and sum(tmp_game._board[x, :] == tmp_game._board[x, 0]) >= n:
                    # return the relative id
                    return tmp_game._board[0, y]
            # if a player has completed the principal diagonal
            if tmp_game._board[0, 0] != -1 and sum(
                [tmp_game._board[x, x]
                    for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, 0]
            )>=n:
                # return the relative id
                return tmp_game._board[0, 0]
            # if a player has completed the secondary diagonal
            if tmp_game._board[0, -1] != -1 and sum(
                [tmp_game._board[x, -(x + 1)]
                for x in range(tmp_game._board.shape[0])] == tmp_game._board[0, -1]
            )>=n:
                # return the relative id
                return tmp_game._board[0, -1]
            return -1
    

    


    def play(tmp_game, player1: Player, player2: Player):
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            tmp_game.current_player_idx += 1
            tmp_game.current_player_idx %= len(players)
            ok = False
            while not ok:
                from_pos, slide = players[tmp_game.current_player_idx].make_move(
                    tmp_game)
                ok = tmp_game._move(from_pos, slide, tmp_game.current_player_idx)
                
     
            
            winner = tmp_game.check_winner()
        return winner

    def play_ql(tmp_game, player1: Player, player2: Player):
        '''Play the game. Returns the winning player'''
        players = [player1, player2]
        winner = -1
        while winner < 0:
            tmp_game.current_player_idx += 1
            tmp_game.current_player_idx %= len(players)
            ok = False
            while not ok:
                from_pos, slide = players[tmp_game.current_player_idx].make_move(
                    tmp_game)
                ok = tmp_game._move(from_pos, slide, tmp_game.current_player_idx)
                
     
            
            winner = tmp_game.check_winner()
        return winner
    def __move(tmp_game, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(tmp_game._board[(from_pos[1], from_pos[0])])
        acceptable = tmp_game.__take((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = tmp_game.__slide((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                tmp_game._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable
    
    def __move_rotation(tmp_game, from_pos: tuple[int, int], slide: Move, player_id: int) -> bool:
        '''Perform a move'''
        if player_id > 2:
            return False
        # Oh God, Numpy arrays
        prev_value = deepcopy(tmp_game._board[(from_pos[1], from_pos[0])])
        acceptable = tmp_game.__take_rotation((from_pos[1], from_pos[0]), player_id)
        if acceptable:
            acceptable = tmp_game.__slide_rotation((from_pos[1], from_pos[0]), slide)
            if not acceptable:
                tmp_game._board[(from_pos[1], from_pos[0])] = deepcopy(prev_value)
        return acceptable

    def __take(tmp_game, from_pos: tuple[int, int], player_id: int) -> bool:
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
        ) and (tmp_game._board[from_pos] < 0 or tmp_game._board[from_pos] == player_id)
        if acceptable:
            tmp_game._board[from_pos] = player_id
        return acceptable
    
    def __accettable(tmp_game, from_pos: tuple[int, int],slide: Move,player_id: int) -> bool:
        '''Look for invalid move on position and direction'''
        ##Look if u can take the piece
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
        ) and (tmp_game._board[from_pos] < 0 or tmp_game._board[from_pos] == player_id)
        if acceptable:
        
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
                return 0
            return -15
            
        
        def __take_rotation(tmp_game, from_pos: tuple[int, int], player_id: int) -> bool:
            '''Take only from the first row and first column because of the rotation'''
            # acceptable only if in border
            acceptable: bool = (
                    # check if it is in the first row
                    (from_pos[0] == 0 and from_pos[1] < 5)
                    # check if it is in the first column
                    or (from_pos[1] == 0 and from_pos[0] < 5)
                    # and check if the piece can be moved by the current player
            ) and (tmp_game._board[from_pos] < 0 or tmp_game._board[from_pos] == player_id)
            if acceptable:
                tmp_game._board[from_pos] = player_id
            return acceptable

        def __slide(tmp_game, from_pos: tuple[int, int], slide: Move) -> bool:
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
                piece = tmp_game._board[from_pos]
                # if the player wants to slide it to the left
                if slide == Move.LEFT:
                    # for each column starting from the column of the piece and moving to the left
                    for i in range(from_pos[1], 0, -1):
                        # copy the value contained in the same row and the previous column
                        tmp_game._board[(from_pos[0], i)] = tmp_game._board[(
                            from_pos[0], i - 1)]
                    # move the piece to the left
                    tmp_game._board[(from_pos[0], 0)] = piece
                # if the player wants to slide it to the right
                elif slide == Move.RIGHT:
                    # for each column starting from the column of the piece and moving to the right
                    for i in range(from_pos[1], tmp_game._board.shape[1] - 1, 1):
                        # copy the value contained in the same row and the following column
                        tmp_game._board[(from_pos[0], i)] = tmp_game._board[(
                            from_pos[0], i + 1)]
                    # move the piece to the right
                    tmp_game._board[(from_pos[0], tmp_game._board.shape[1] - 1)] = piece
                # if the player wants to slide it upward
                elif slide == Move.TOP:
                    # for each row starting from the row of the piece and going upward
                    for i in range(from_pos[0], 0, -1):
                        # copy the value contained in the same column and the previous row
                        tmp_game._board[(i, from_pos[1])] = tmp_game._board[(
                            i - 1, from_pos[1])]
                    # move the piece .
                        
                        
                    tmp_game._board[(0, from_pos[1])] = piece
                # if the player wants to slide it downward
                elif slide == Move.BOTTOM:
                    # for each row starting from the row of the piece and going downward
                    for i in range(from_pos[0], tmp_game._board.shape[0] - 1, 1):
                        # copy the value contained in the same column and the following row
                        tmp_game._board[(i, from_pos[1])] = tmp_game._board[(
                            i + 1, from_pos[1])]
                    # move the piece down
                    tmp_game._board[(tmp_game._board.shape[0] - 1, from_pos[1])] = piece
            return acceptable
    
    def __slide_rotation(tmp_game, from_pos: tuple[int, int], slide: Move) -> bool:
        '''Slide the other pieces'''
        # define the corners
        SIDES = [(0, 0), (0, 4), (4, 0), (4, 4)]
        # if the piece position is not in a corner
        if from_pos not in SIDES:
            # if it is at the TOP, it can be moved down, left or right
            acceptable_top: bool = from_pos[0] == 0 and (
                slide == Move.BOTTOM or slide == Move.LEFT or slide == Move.RIGHT
            )
            # if it is on the LEFT, it can be moved up, down or right
            acceptable_left: bool = from_pos[1] == 0 and (
                slide == Move.BOTTOM or slide == Move.TOP or slide == Move.RIGHT
            )
            acceptable_right:bool = False
            
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
    
        # check if the move is acceptable
        acceptable: bool = acceptable_top or acceptable_left or acceptable_right
        # if it is
        if acceptable:
            # take the piece
            piece = tmp_game._board[from_pos]
            # if the player wants to slide it to the left
            if slide == Move.LEFT:
                # for each column starting from the column of the piece and moving to the left
                for i in range(from_pos[1], 0, -1):
                    # copy the value contained in the same row and the previous column
                    tmp_game._board[(from_pos[0], i)] = tmp_game._board[(
                        from_pos[0], i - 1)]
                # move the piece to the left
                tmp_game._board[(from_pos[0], 0)] = piece
            # if the player wants to slide it to the right
            elif slide == Move.RIGHT:
                # for each column starting from the column of the piece and moving to the right
                for i in range(from_pos[1], tmp_game._board.shape[1] - 1, 1):
                    # copy the value contained in the same row and the following column
                    tmp_game._board[(from_pos[0], i)] = tmp_game._board[(
                        from_pos[0], i + 1)]
                # move the piece to the right
                tmp_game._board[(from_pos[0], tmp_game._board.shape[1] - 1)] = piece
            # if the player wants to slide it upward
            elif slide == Move.TOP:
                # for each row starting from the row of the piece and going upward
                for i in range(from_pos[0], 0, -1):
                    # copy the value contained in the same column and the previous row
                    tmp_game._board[(i, from_pos[1])] = tmp_game._board[(
                        i - 1, from_pos[1])]
                # move the piece .
                    
                    
                tmp_game._board[(0, from_pos[1])] = piece
            # if the player wants to slide it downward
            elif slide == Move.BOTTOM:
                # for each row starting from the row of the piece and going downward
                for i in range(from_pos[0], tmp_game._board.shape[0] - 1, 1):
                    # copy the value contained in the same column and the following row
                    tmp_game._board[(i, from_pos[1])] = tmp_game._board[(
                        i + 1, from_pos[1])]
                # move the piece down
                tmp_game._board[(tmp_game._board.shape[0] - 1, from_pos[1])] = piece
        return acceptable
    
    def rotate_multiple_matrix(tmp_game):
        '''Return the board with different rotation such as  original_board,rotate_board_90, rotate_board_180, rotate_board_270 '''
        row = len(tmp_game._board)
        col = len(tmp_game._board[0])

        # Creazione di una nuova matrix vuota con col e row invertite
        rotate_matrix_90 = [[0] * row for _ in range(col)]
        rotate_matrix_180 = [[0] * row for _ in range(col)]
        rotate_matrix_270 = [[0] * row for _ in range(col)]

        # Applicazione delle rotazioni
        for i in range(row):
            for j in range(col):
                rotate_matrix_90[j][row - 1 - i] = tmp_game._board[i][j]
                rotate_matrix_180[row - 1 - i][col - 1 - j] = tmp_game._board[i][j]
                rotate_matrix_270[col - 1 - j][i] = tmp_game._board[i][j]
        
        return [tmp_game._board,rotate_matrix_90, rotate_matrix_180, rotate_matrix_270]



    def  compute_reward(tmp_game,from_pos,slide) -> int :
       '''Return  the sum of  maximium between positive rewards and the minimium between the negative one's  '''
       if tmp_game.check_2_pattern(from_pos,slide):
        reward2= 0.2;
       if tmp_game.check_3_pattern(from_pos,slide):
        reward3= 0.3;
       if tmp_game.check_4_pattern(from_pos,slide):
        reward4= 0.4;
       if tmp_game.check_is_winning(from_pos,slide):
        reward5=0.5;
       reward6=tmp_game.__accettable(tmp_game,from_pos,slide,1);
       
       return max(reward2,reward3,reward4,reward5)+min(reward6,-0.4)

       
