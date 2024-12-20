import chess
import pygame
import random
import time
from bot import ChessBot

# Color themes
# LIGHT_SQUARE = (220, 230, 240)                # Pale blue-gray
# DARK_SQUARE = (93, 109, 126)                  # Steel blue
# LIGHT_SQUARE = (238, 238, 210)                # Cream
# DARK_SQUARE = (118, 150, 86)                  # Forest green
# LIGHT_SQUARE = (232, 220, 240)                # Light lavender
# DARK_SQUARE = (95, 75, 139)                   # Royal purple
# LIGHT_SQUARE = (209, 231, 240)                # Sky blue
# DARK_SQUARE = (69, 119, 142)                  # Ocean blue
LIGHT_SQUARE = (164, 164, 164)                  # Silver gray
DARK_SQUARE = (87, 87, 87)                      # Charcoal
# LIGHT_SQUARE = (210, 175, 160)                # Warm beige
# DARK_SQUARE = (90, 50, 45)                    # Deep maroon

# HIGHLIGHT = (130, 151, 105, 150)
# CHECK_HIGHLIGHT = (232, 46, 46, 150)
# CAPTURE_HIGHLIGHT = (255, 116, 108, 150)
HIGHLIGHT = (150, 190, 100, 150)              # Muted green
CHECK_HIGHLIGHT = (220, 50, 50, 150)          # Crimson
CAPTURE_HIGHLIGHT = (250, 130, 100, 150)      # Soft red-orange
LAST_MOVE_HIGHLIGHT = (255, 255, 150, 150)    # Soft yellow
MOVE_DOT = (255, 255, 255, 125)               # Semi-transparent white


# Initialize pygame
WINDOW_SIZE = 800
SQUARE_SIZE = WINDOW_SIZE // 8
PIECE_IMAGES = {}
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("Chess")

# Loads piece images
def load_pieces():
    pieces = ['p', 'n', 'b', 'r', 'q', 'k']
    for piece in pieces:
        original = pygame.image.load(f'assets/w{piece}.png')
        original = original.convert_alpha()
        PIECE_IMAGES[f'w{piece}'] = pygame.transform.smoothscale(
            original,
            (SQUARE_SIZE, SQUARE_SIZE)
        )
        original = pygame.image.load(f'assets/b{piece}.png')
        original = original.convert_alpha()
        PIECE_IMAGES[f'b{piece}'] = pygame.transform.smoothscale(
            original,
            (SQUARE_SIZE, SQUARE_SIZE)
        )

# Converts mouse position to board square
def get_square_from_mouse(pos, flip=False):
    x, y = pos
    if flip:
        x = WINDOW_SIZE - x
        y = WINDOW_SIZE - y
    return chess.Square(x // SQUARE_SIZE + 8 * (7 - y // SQUARE_SIZE))

# Converts board square to screen coordinates
def get_screen_coords(square, flip=False):
    file_idx = chess.square_file(square)
    rank_idx = 7 - chess.square_rank(square)
    if flip:
        file_idx = 7 - file_idx
        rank_idx = 7 - rank_idx
    return (file_idx * SQUARE_SIZE, rank_idx * SQUARE_SIZE)

# Draws the chess board checker pattern
def draw_board(flip=False):
    for rank in range(8):
        for file in range(8):
            color = LIGHT_SQUARE if (rank + file) % 2 == 0 else DARK_SQUARE
            if flip:
                file = 7 - file
                rank = 7 - rank
            pygame.draw.rect(screen, color,
                           (file * SQUARE_SIZE, rank * SQUARE_SIZE,
                            SQUARE_SIZE, SQUARE_SIZE))

# Draw the chess pieces on the board
def draw_pieces(board, flip=False):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color else 'b'
            piece_img = PIECE_IMAGES[f'{color}{piece.symbol().lower()}']
            x, y = get_screen_coords(square, flip)
            screen.blit(piece_img, (x, y))

# Draw highlights for selected piece, valid moves, and check
def draw_highlights(selected_square, valid_moves, board, last_move=None, flip=False):
    # The selected piece
    if selected_square is not None:
        x, y = get_screen_coords(selected_square, flip)
        highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(highlight_surf, HIGHLIGHT, highlight_surf.get_rect())
        screen.blit(highlight_surf, (x, y))

    # The move indicators
    for move in valid_moves:
        x, y = get_screen_coords(move.to_square, flip)
        
        if board.piece_at(move.to_square):
            # For captures, draw a larger hollow circle
            radius = SQUARE_SIZE // 2.25
            circle_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, CAPTURE_HIGHLIGHT, 
                             (SQUARE_SIZE//2, SQUARE_SIZE//2), radius, 4,
                             draw_top_right=True, draw_top_left=True,
                             draw_bottom_right=True, draw_bottom_left=True)
            screen.blit(circle_surf, (x, y))
        else:
            # For regular moves, draw a small filled circle
            radius = SQUARE_SIZE // 4
            circle_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, MOVE_DOT,
                             (SQUARE_SIZE//2, SQUARE_SIZE//2), radius,
                             draw_top_right=True, draw_top_left=True,
                             draw_bottom_right=True, draw_bottom_left=True)
            screen.blit(circle_surf, (x, y))

    # The king in check
    if board.is_check():
        king_square = chess.square(0, 0)
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece and piece.piece_type == chess.KING and piece.color == board.turn:
                king_square = square
                break
        
        x, y = get_screen_coords(king_square, flip)
        highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(highlight_surf, CHECK_HIGHLIGHT, highlight_surf.get_rect())
        screen.blit(highlight_surf, (x, y))

    # Add last move highlight
    if last_move:
        for square in [last_move.from_square, last_move.to_square]:
            x, y = get_screen_coords(square, flip)
            highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surf, LAST_MOVE_HIGHLIGHT, highlight_surf.get_rect())
            screen.blit(highlight_surf, (x, y))

# Shows interface for pawn promotion
def show_promotion_selection(flip=False, player_color=chess.WHITE):
    promotion_pieces = [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]
    piece_symbols = ['q', 'r', 'b', 'n']
    
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 150))
    screen.blit(overlay, (0, 0))

    box_size = SQUARE_SIZE
    start_x = (WINDOW_SIZE - (box_size * 4)) // 2
    start_y = (WINDOW_SIZE - box_size) // 2

    for i, symbol in enumerate(piece_symbols):
        x = start_x + (i * box_size)
        pygame.draw.rect(screen, LIGHT_SQUARE, (x, start_y, box_size, box_size))
        pygame.draw.rect(screen, DARK_SQUARE, (x, start_y, box_size, box_size), 2)
        
        piece_img = PIECE_IMAGES[f'{"w" if player_color else "b"}{symbol}']  # Use player_color here
        screen.blit(piece_img, (x, start_y))
    
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if start_y <= y <= start_y + box_size:
                    selection = (x - start_x) // box_size
                    if 0 <= selection < 4:
                        return promotion_pieces[selection]

# Choose player color: white, black, or random
def choose_color():
    font = pygame.font.Font('assets/caviar_dreams.ttf', WINDOW_SIZE // 16)
    text = "Choose your color: W/B/R"
    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2))
    bg = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE))
    bg.set_alpha(200)
    draw_board()
    screen.blit(bg, (0, 0))
    screen.blit(text_surface, text_rect)
    pygame.display.flip()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_w:
                    return chess.WHITE
                elif event.key == pygame.K_b:
                    return chess.BLACK
                elif event.key == pygame.K_r:
                    return random.choice([chess.WHITE, chess.BLACK])

# Gets called during alpha beta search to avoid the window from not repsonding        
def handle_events_during_bot_think():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
    pygame.display.flip()

# Da main function: int main(void) { /* Cool Code */ return 0; } hehe
def main():
    load_pieces()
    board = chess.Board()
    selected_square = None
    valid_moves = []
    game_over = False
    last_move = None
    bot = ChessBot()
    bot.thinking_callback = handle_events_during_bot_think # Add callback to bot instance

    player_color = choose_color()
    running = True
    font_s = pygame.font.Font('assets/caviar_dreams.ttf', 32)
    font_l = pygame.font.Font('assets/caviar_dreams.ttf', 64)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_SPACE:
                    board = chess.Board()
                    bot.reset()
                    selected_square = None
                    valid_moves = []
                    game_over = False
                    player_color = choose_color()  # Prompt to choose color again

            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if event.button == 1 and board.turn == player_color:  # Left click and player's turn
                    flip = player_color == chess.BLACK
                    clicked_square = get_square_from_mouse(event.pos, flip)
                    
                    if selected_square is None:
                        piece = board.piece_at(clicked_square)
                        if piece and piece.color == board.turn:
                            selected_square = clicked_square
                            valid_moves = [move for move in board.legal_moves 
                                         if move.from_square == clicked_square]
                    else:
                        move = chess.Move(selected_square, clicked_square)

                        # Check if this move exists in valid moves (either as is or as a promotion)
                        matching_moves = [m for m in valid_moves 
                                        if m.from_square == selected_square 
                                        and m.to_square == clicked_square]
                        
                        if matching_moves:
                            # If it's a pawn reaching the back rank
                            piece = board.piece_at(selected_square)
                            is_promotion = (piece 
                                          and piece.piece_type == chess.PAWN 
                                          and ((clicked_square >= chess.A8 and piece.color) 
                                               or (clicked_square <= chess.H1 and not piece.color)))
                            
                            if is_promotion:
                                promotion_piece = show_promotion_selection(flip, board.turn)
                                move = chess.Move(selected_square, clicked_square, promotion_piece)
                            else:
                                move = matching_moves[0]  # Use the first matching move
                                
                            if move in board.legal_moves:
                                board.push(move)
                                last_move = move
                                
                        selected_square = None
                        valid_moves = []

        # Draw stuff
        flip = player_color == chess.BLACK
        draw_board(flip)
        draw_highlights(selected_square, valid_moves, board, last_move, flip)
        draw_pieces(board, flip)

        if game_over:
            # show the game over screen
            overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 100))
            screen.blit(overlay, (0, 0))

            if board.is_checkmate():
                text = "Checkmate! {} wins!".format("Black" if board.turn == chess.WHITE else "White")
            elif board.is_stalemate():
                text = "Stalemate!"
            else:
                text = "Draw!"

            text_surface = font_l.render(text, True, (250, 250, 250))
            text_rect = text_surface.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 - 50))
            screen.blit(text_surface, text_rect)

            restart_text = "Press SPACE to play again"
            restart_surface = font_s.render(restart_text, True, (250, 250, 250))
            restart_rect = restart_surface.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 + 50))
            screen.blit(restart_surface, restart_rect)
        elif board.turn != player_color:
            # Show the "Thinking..." message
            thinking_text = font_s.render("Thinking...", True, (250, 250, 250))
            thinking_rect = thinking_text.get_rect(center=(WINDOW_SIZE/2, 50))
            bg = pygame.Surface((WINDOW_SIZE, 100))
            bg.fill((0, 0, 0))
            bg.set_alpha(150)
            screen.blit(bg, (0, 0))
            screen.blit(thinking_text, thinking_rect)
            
            # bot's move
            if not board.is_checkmate() and not board.is_stalemate():
                move = bot.think(board, timer={
                                    'start': time.time(),
                                    'time': 15,  # 15 seconds
                                })
                board.push(move)
                last_move = move

        if board.is_game_over() and not game_over:
            # reset the game
            game_over = True

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()