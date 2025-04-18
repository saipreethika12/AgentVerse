import chess
import math

# Piece values for evaluation
PIECE_VALUES = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0  # Not counted normally, checkmate is handled separately
}

# Evaluation function


def evaluate_board(board):
    if board.is_checkmate():
        return -9999 if board.turn else 9999
    elif board.is_stalemate() or board.is_insufficient_material():
        return 0

    value = 0
    for piece_type in PIECE_VALUES:
        value += len(board.pieces(piece_type, chess.WHITE)) * \
            PIECE_VALUES[piece_type]
        value -= len(board.pieces(piece_type, chess.BLACK)) * \
            PIECE_VALUES[piece_type]
    return value

# Minimax with Alpha-Beta Pruning


def minimax(board, depth, alpha, beta, maximizing_player):
    if depth == 0 or board.is_game_over():
        return evaluate_board(board), None

    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, False)
            board.pop()

            if eval > max_eval:
                max_eval = eval
                best_move = move

            alpha = max(alpha, eval)
            if beta <= alpha:  # Alpha-Beta Pruning
                break
        return max_eval, best_move

    else:
        min_eval = math.inf
        for move in board.legal_moves:
            board.push(move)
            eval, _ = minimax(board, depth - 1, alpha, beta, True)
            board.pop()

            if eval < min_eval:
                min_eval = eval
                best_move = move

            beta = min(beta, eval)
            if beta <= alpha:  # Alpha-Beta Pruning
                break
        return min_eval, best_move


# Simulate a game using Alpha-Beta Pruning
board = chess.Board()

while not board.is_game_over():
    print(board.unicode())
    print("Turn:", "White" if board.turn else "Black")

    eval, best_move = minimax(
        board, depth=3, alpha=-math.inf, beta=math.inf, maximizing_player=board.turn)
    print("Best move:", best_move)
    board.push(best_move)

print("Game over!")
print(board.result())
