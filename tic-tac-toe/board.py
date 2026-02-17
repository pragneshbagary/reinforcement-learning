class Board:

    def make_a_move(self, board, idx, val):
        temp = list(board)

        temp[idx] = val
        new_state = tuple(temp)
        
        return new_state
    
    def get_legal_moves(self, board):
        temp = []
        for i in range(9):
            if(board[i] == '.'):
                temp.append(i)

        return temp
    
    def print_board(self, board):
        for i in range(0, 9, 3):
            print(f" {board[i]} | {board[i+1]} | {board[i+2]} ")
            if i < 6:
                print("---+---+---")

    def check_winner(self, board):
        win_conditions = [
            (0, 1, 2),  # row 1
            (3, 4, 5),  # row 2
            (6, 7, 8),  # row 3
            (0, 3, 6),  # col 1
            (1, 4, 7),  # col 2
            (2, 5, 8),  # col 3
            (0, 4, 8),  # diagonal
            (2, 4, 6),  # anti-diagonal
        ]

        for a, b, c in win_conditions:
            if board[a] == board[b] == board[c] and board[a] != '.':
                return board[a]  # returns 'X' or 'O'
        return None
    
    def check_draw(self, board):
        return '.' not in board


