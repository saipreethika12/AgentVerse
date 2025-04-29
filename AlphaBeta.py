import gym
import gym_chess
import chess

# --- Evaluation Function ---
def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    if board.is_stalemate() or board.is_insufficient_material():
        return 0

    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }

    # Piece-square tables (truncated for brevity; use full ones for better performance)
    pawn_table = [
         0,  5,  5,-10,-10,  5,  5,  0,
         0, 10,-10,  0,  0,-10, 10,  0,
         0, 10, 10, 20, 20, 10, 10,  0,
         5, 20, 20, 30, 30, 20, 20,  5,
        10, 20, 20, 30, 30, 20, 20, 10,
         5, 10, 10, 20, 20, 10, 10,  5,
         0,  0,  0,  0,  0,  0,  0,  0,
         0,  0,  0,  0,  0,  0,  0,  0
    ]

    knight_table = [
        -50,-40,-30,-30,-30,-30,-40,-50,
        -40,-20,  0,  5,  5,  0,-20,-40,
        -30,  5, 10, 15, 15, 10,  5,-30,
        -30,  0, 15, 20, 20, 15,  0,-30,
        -30,  5, 15, 20, 20, 15,  5,-30,
        -30,  0, 10, 15, 15, 10,  0,-30,
        -40,-20,  0,  0,  0,  0,-20,-40,
        -50,-40,-30,-30,-30,-30,-40,-50,
    ]

    queen_table = [
        -20,-10,-10, -5, -5,-10,-10,-20,
        -10,  0,  0,  0,  0,  0,  0,-10,
        -10,  0,  5,  5,  5,  5,  0,-10,
         -5,  0,  5,  5,  5,  5,  0, -5,
          0,  0,  5,  5,  5,  5,  0, -5,
        -10,  5,  5,  5,  5,  5,  0,-10,
        -10,  0,  5,  0,  0,  0,  0,-10,
        -20,-10,-10, -5, -5,-10,-10,-20
    ]

    value = 0
    white_bishop_count = 0
    black_bishop_count = 0

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if not piece:
            continue
        val = piece_values[piece.piece_type]

        # Positional bonuses
        index = square if piece.color == chess.WHITE else chess.square_mirror(square)
        if piece.piece_type == chess.PAWN:
            val += pawn_table[index]
        elif piece.piece_type == chess.KNIGHT:
            val += knight_table[index]
        elif piece.piece_type == chess.QUEEN:
            val += queen_table[index]

        # Bishop pair bonus
        if piece.piece_type == chess.BISHOP:
            if piece.color == chess.WHITE:
                white_bishop_count += 1
            else:
                black_bishop_count += 1

        # Rook file bonus
        if piece.piece_type == chess.ROOK:
            file_empty = True
            for r in range(8):
                sq = chess.square(chess.square_file(square), r)
                occupant = board.piece_at(sq)
                if occupant and occupant.piece_type == chess.PAWN:
                    file_empty = False
                    break
            if file_empty:
                val += 10

        value += val if piece.color == chess.WHITE else -val

    # Bishop pair bonus
    if white_bishop_count >= 2:
        value += 50
    if black_bishop_count >= 2:
        value -= 50

    # Mobility bonus
    board_turn = board.turn
    white_mobility = len(list(board.legal_moves)) if board_turn == chess.WHITE else 0
    board.push(chess.Move.null())
    black_mobility = len(list(board.legal_moves)) if board.turn == chess.BLACK else 0
    board.pop()

    value += (white_mobility - black_mobility) * 10

    return value



# --- Alpha-Beta Pruning ---
def alpha_beta(board, depth, alpha, beta, maximizing):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    legal_moves = list(board.legal_moves)

    if maximizing:
        max_eval = float('-inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alpha_beta(board, depth - 1, alpha, beta, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in legal_moves:
            board.push(move)
            eval, _ = alpha_beta(board, depth - 1, alpha, beta, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# --- Main Execution Function ---
def play_game_with_alpha_beta(depth=3):
    env = gym.make('Chess-v0')
    env.reset()
    board = chess.Board()

    print("Initial Board:")
    print(board.unicode())

    done = False
    move_count = 1

    while not done:
        # White Move
        if board.turn == chess.WHITE:
            _, move = alpha_beta(board, depth, float('-inf'), float('inf'), maximizing=True)
            if move is None:
                break
            board.push(move)
            env.step(move)
            print(f"\nMove {move_count} White: {move.uci()}")
            print(board.unicode())
            if board.is_game_over():
                break

        # Black Move
        if board.turn == chess.BLACK:
            _, move = alpha_beta(board, depth, float('-inf'), float('inf'), maximizing=False)
            if move is None:
                break
            board.push(move)
            env.step(move)
            print(f"\nMove {move_count} Black: {move.uci()}")
            print(board.unicode())
            if board.is_game_over():
                break

        move_count += 1

    print("\nGame Over!")
    print("Result:", board.result())


# --- Run the game ---
if __name__ == "__main__":
    play_game_with_alpha_beta(depth=3)

