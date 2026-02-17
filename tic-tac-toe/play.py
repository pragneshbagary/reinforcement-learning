from board import Board
from agent import Agent
import pickle

def display_board(state):
    """Display the board in a readable format"""
    board_map = {idx: cell for idx, cell in enumerate(state)}
    print("\n")
    for i in range(0, 9, 3):
        row = []
        for j in range(3):
            idx = i + j
            cell = board_map[idx]
            if cell == '.':
                row.append(str(idx))
            else:
                row.append(cell)
        print(f" {row[0]} | {row[1]} | {row[2]} ")
        if i < 6:
            print("---+---+---")
    print()

def load_agent():
    """Load the trained agent from file"""
    try:
        with open('trained_agent.pkl', 'rb') as f:
            agent = pickle.load(f)
        agent.epsilon = 0  # No exploration during play
        return agent
    except FileNotFoundError:
        print("Error: trained_agent.pkl not found.")
        print("Please run 'python train.py' first to train the agent.")
        exit()

def play_game():
    """Interactive game loop"""
    board = Board()
    agent = load_agent()
    
    state = tuple('.' * 9)
    current_player = 'X'  # Human is X
    
    print("=" * 40)
    print("Tic-Tac-Toe vs Trained Agent")
    print("=" * 40)
    print("\nYou are X, Agent is O")
    print("Enter position (0-8) to make your move:\n")
    display_board(state)
    
    while True:
        if current_player == 'X':
            # Human player
            while True:
                try:
                    move = int(input("Your move: "))
                    legal_moves = board.get_legal_moves(state)
                    
                    if move not in legal_moves:
                        print(f"Invalid move. Available positions: {legal_moves}")
                        continue
                    break
                except ValueError:
                    print("Please enter a number between 0 and 8.")
            
            state = board.make_a_move(state, move, 'X')
        else:
            # Agent player
            legal_moves = board.get_legal_moves(state)
            if not legal_moves:
                break
            
            action = agent.choose_action(state, legal_moves, board)
            print(f"Agent moves to position {action}")
            state = board.make_a_move(state, action, 'O')
        
        display_board(state)
        
        # Check for winner
        winner = board.check_winner(state)
        if winner:
            if winner == 'X':
                print("Congratulations! You won!")
            else:
                print("Agent won! Better luck next time.")
            break
        
        # Check for draw
        if board.check_draw(state):
            print("It's a draw!")
            break
        
        # Switch player
        current_player = 'O' if current_player == 'X' else 'X'

if __name__ == "__main__":
    play_game()
