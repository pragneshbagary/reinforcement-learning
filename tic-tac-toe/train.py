from board import Board
from agent import Agent
import random

N = 10000
board = Board()
agent = Agent()

X_wins = 0
O_wins = 0
draws = 0
total_reward = 0


episode_X_wins = 0
episode_O_wins = 0
episode_draws = 0
episode_reward = 0

for episode in range(1,N+1):

   

    state = tuple('.' * 9)
    current_player = 'X'
    episode_list = []



    while(True):

        legal_moves = board.get_legal_moves(state)
        if current_player == 'X':
            action = agent.choose_action(state, legal_moves, board)
        else:
            action = random.choice(legal_moves)

        state = board.make_a_move(state, action, current_player)
        if current_player == 'X' :
            episode_list.append(state)

        winner = board.check_winner(state)
        if winner == 'X':
            agent.update(episode_list, 1)
            total_reward = total_reward + 1
            episode_reward = episode_reward + 1
            X_wins = X_wins + 1
            episode_X_wins = episode_X_wins + 1
            # print(f"winner is {winner}")
            # board.print_board(state)
            break 
        elif winner == 'O':
            agent.update(episode_list, 0)
            O_wins = O_wins + 1
            episode_O_wins = episode_O_wins + 1
            # print(f"winner is {winner}")
            # board.print_board(state)
            break 

        if board.check_draw(state):
            agent.update(episode_list, 0.5)
            total_reward = total_reward + 0.5
            episode_reward = episode_reward + 0.5
            draws = draws + 1
            episode_draws = episode_draws + 1
            # print("game draw!!!")
            # print(state)
            break

        
        current_player = 'O' if current_player == 'X' else 'X'

    if episode % 1000 == 0:
        print(f"--- Episode {episode} (last 1000) ---")
        print(f"X wins = {episode_X_wins}")
        print(f"O wins = {episode_O_wins}")
        print(f"Draws = {episode_draws}")
        print(f"Average Reward = {episode_reward/1000:.4f}")

        episode_X_wins = 0
        episode_O_wins = 0
        episode_draws = 0
        episode_reward = 0

    if episode % 10000 == 0:
        print("#" * 50)
        board.print_board(state)
        print(f"--- Episode {episode} (total) ---")
        print(f"Total X wins = {X_wins}")
        print(f"Total O wins = {O_wins}")
        print(f"Total Draws = {draws}")
        print(f"Average Reward = {total_reward/episode:.4f}")
        print("#" * 50)

# ---- EVALUATION ----
agent.epsilon = 0   

test_games = 1000

X_wins = 0
O_wins = 0
draws = 0

for _ in range(test_games):

    state = tuple('.' * 9)
    current_player = 'X'

    while True:

        legal_moves = board.get_legal_moves(state)

        if current_player == 'X':
            action = agent.choose_action(state, legal_moves, board)
        else:
            action = random.choice(legal_moves)

        state = board.make_a_move(state, action, current_player)

        winner = board.check_winner(state)

        if winner == 'X':
            X_wins += 1
            break

        elif winner == 'O':
            O_wins += 1
            break

        elif board.check_draw(state):
            draws += 1
            break

        current_player = 'O' if current_player == 'X' else 'X'


print("\n===== Evaluation Results =====")
print(f"X wins: {X_wins}")
print(f"O wins: {O_wins}")
print(f"Draws: {draws}")
print(f"Win rate: {X_wins / test_games:.3f}")
print(f"Loss rate: {O_wins / test_games:.3f}")
print(f"Draw rate: {draws / test_games:.3f}")
