from board import Board
from agent import Agent
import random
import pickle

N = 100000
board = Board()
agent_one = Agent()
agent_two = Agent()

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
    episode_list_O = []
    episode_list_X = []



    while(True):

        legal_moves = board.get_legal_moves(state)
        if current_player == 'X':
            action = agent_one.choose_action(state, legal_moves, board, current_player)
        else:
            action = agent_two.choose_action(state, legal_moves, board, current_player)

        state = board.make_a_move(state, action, current_player)
        if current_player == 'O' :
            episode_list_O.append(state)
        else:
            episode_list_X.append(state)

        winner = board.check_winner(state)
        if winner == 'O':
            agent_two.update(episode_list_O , 1)
            agent_one.update(episode_list_X, 0)
            total_reward = total_reward + 1
            episode_reward = episode_reward + 1
            O_wins = O_wins + 1
            episode_O_wins = episode_O_wins + 1
            # print(f"winner is {winner}")
            # board.print_board(state)
            break 
        elif winner == 'X':
            agent_one.update(episode_list_X, 1)
            agent_two.update(episode_list_O, 0)
            X_wins = X_wins + 1
            episode_X_wins = episode_X_wins + 1
            # print(f"winner is {winner}")
            # board.print_board(state)
            break 

        if board.check_draw(state):
            agent_one.update(episode_list_X, 0.5)
            agent_two.update(episode_list_O, 0.5)
            total_reward = total_reward + 0.5
            episode_reward = episode_reward + 0.5
            draws = draws + 1
            episode_draws = episode_draws + 1
            # print("game draw!!!")
            # print(state)
            break

        
        current_player = 'O' if current_player == 'X' else 'X'

    if episode % 10000 == 0:
        print(f"--- Episode {episode} (last 1000) ---")
        print(f"X wins = {episode_X_wins}")
        print(f"O wins = {episode_O_wins}")
        print(f"Draws = {episode_draws}")
        print(f"Average Reward = {episode_reward/1000:.4f}")

        episode_X_wins = 0
        episode_O_wins = 0
        episode_draws = 0
        episode_reward = 0

    if episode % 100000 == 0:
        print("#" * 50)
        board.print_board(state)
        print(f"--- Episode {episode} (total) ---")
        print(f"Total X wins = {X_wins}")
        print(f"Total O wins = {O_wins}")
        print(f"Total Draws = {draws}")
        print(f"Average Reward = {total_reward/episode:.4f}")
        print("#" * 50)

# ---- EVALUATION ----
agent_one.epsilon = 0   
agent_two.epsilon = 0

test_games = 1000

X_wins = 0
O_wins = 0
draws = 0

for _ in range(test_games):

    state = tuple('.' * 9)
    current_player = 'X'

    while True:

        legal_moves = board.get_legal_moves(state)

        if current_player == 'O':
            action = agent_one.choose_action(state, legal_moves, board, current_player) 
        else:
            action = agent_two.choose_action(state, legal_moves, board, current_player)

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
print(f"Win rate: {O_wins / test_games:.3f}")
print(f"Loss rate: {X_wins / test_games:.3f}")
print(f"Draw rate: {draws / test_games:.3f}")

# Save the trained agent
# with open('trained_agent.pkl', 'wb') as f:
#     pickle.dump(agent, f)
# print("\nAgent saved to trained_agent.pkl")
