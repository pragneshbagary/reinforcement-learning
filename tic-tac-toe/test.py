
# ---- EVALUATION ----
agent.epsilon = 0   # turn off exploration

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
            # Greedy move (epsilon = 0)
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
