import gym
import gym_chess
import chess

# Evaluation function: simple material count
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


# Minimax algorithm
def minimax(board, depth, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None
    if maximizing_player:
        max_eval = float('-inf')
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, False)
            board.pop()
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, True)
            board.pop()
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

def play_game(depth=3):
    env = gym.make('Chess-v0')
    env.reset()
    board = chess.Board()

    print("Initial Board:")
    print(board.unicode())

    move_number = 1

    while not board.is_game_over():
        if board.turn == chess.WHITE:
            # Minimax plays as White
            _, move = minimax(board, depth, True)
            print(f"Move {move_number} White: {move}")
        else:
            # Minimax plays as Black
            _, move = minimax(board, depth, False)
            print(f"Move {move_number} Black: {move}")
            move_number+=1

        board.push(move)
        env.step(move)
        print(board.unicode())
        print("-" * 30)

    print("Game Over")
    print(f"Result: {board.result()}")


if __name__ == "__main__":
    play_game(depth=3)