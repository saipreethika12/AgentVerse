import gym
import gym_chess
import chess
import time
import chess.svg
import cairosvg
import io
from PIL import ImageDraw, ImageFont, Image


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


def board_to_image(board, move_number=None):
    svg_data = chess.svg.board(board=board)
    png_bytes = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
    image = Image.open(io.BytesIO(png_bytes)).convert("RGB")

    if move_number is not None:
        draw = ImageDraw.Draw(image)
        try:
            font = ImageFont.truetype("arial.ttf", size=24)
        except:
            font = ImageFont.load_default()

        text = f"Move {move_number}"
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        image_width, image_height = image.size
        position = (15, image_height - text_height - 25)

        draw.text(position, text, fill="black", font=font)


    return image

def create_gif_from_images(images, output_path, duration=500):
    images[0].save(output_path, save_all=True, append_images=images[1:], duration=duration, loop=0)
    
def final_board_with_stats(board, result, move_count):
    image = board_to_image(board)  # get the final board image

    draw = ImageDraw.Draw(image)
    try:
        big_font = ImageFont.truetype("arial.ttf", size=48)
        small_font = ImageFont.truetype("arial.ttf", size=28)
    except:
        big_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Text values
    result_text = f"Result: {result}"
    moves_text = f"Moves: {move_count}"

    # Positioning
    width, height = image.size
    spacing = 40

    draw.text((50, 150), result_text, fill="black", font=big_font)
    draw.text((130, 250), moves_text, fill="black", font=small_font)

    return image



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
    move_count = 0
    start_time = time.time()
    images=[]
    images.append(board_to_image(board,move_count))

    while not done:
        # White Move
        if board.turn == chess.WHITE: move_count += 1
        if board.turn == chess.WHITE:
            _, move = alpha_beta(board, depth, float('-inf'), float('inf'), maximizing=True)
            if move is None:
                break
            board.push(move)
            env.step(move)
            print(f"\nMove {move_count} White: {move.uci()}")
            print(board.unicode())
            images.append(board_to_image(board,move_count))
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
            images.append(board_to_image(board,move_count))
            print("-" * 30)
            if board.is_game_over():
                break


    end_time = time.time()
    total_time = end_time - start_time
    for i in range(10): images.append(final_board_with_stats(board, board.result(), move_count))
    print("Game Over")
    print(f"Result: {board.result()}")
    print("Algoritm used: Alpha-Beta Pruning")
    print(f"Depth: {depth}")
    print(f"Total Moves: {move_count}")
    print(f"Total Time Taken: {total_time:.2f} seconds")
    create_gif_from_images(images,f"chess_game_depth_{depth}.gif")
    print(f"GIF saved as 'chess_game_depth_{depth}.gif'")



# --- Run the game ---
if __name__ == "__main__":
    play_game_with_alpha_beta(depth=4)

