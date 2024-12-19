import chess
import pygame
import os
import random
import time
from bot import ChessBot

# color themes
# LIGHT_SQUARE = (220, 230, 240)                # pale blue-gray
# DARK_SQUARE = (93, 109, 126)                  # steel blue
# LIGHT_SQUARE = (238, 238, 210)                # cream
# DARK_SQUARE = (118, 150, 86)                  # forest green
# LIGHT_SQUARE = (232, 220, 240)                # light lavender
# DARK_SQUARE = (95, 75, 139)                   # royal purple
# LIGHT_SQUARE = (209, 231, 240)                # sky blue
# DARK_SQUARE = (69, 119, 142)                  # ocean blue
LIGHT_SQUARE = (164, 164, 164)                # silver gray
DARK_SQUARE = (87, 87, 87)                    # charcoal
# LIGHT_SQUARE = (210, 175, 160)                # warm beige
# DARK_SQUARE = (90, 50, 45)                    # deep maroon

# HIGHLIGHT = (130, 151, 105, 150)
# CHECK_HIGHLIGHT = (232, 46, 46, 150)
# CAPTURE_HIGHLIGHT = (255, 116, 108, 150)
HIGHLIGHT = (150, 190, 100, 150)              # muted green
CHECK_HIGHLIGHT = (220, 50, 50, 150)          # crimson
CAPTURE_HIGHLIGHT = (250, 130, 100, 150)      # soft red-orange
LAST_MOVE_HIGHLIGHT = (255, 255, 150, 150)    # soft yellow
MOVE_DOT = (255, 255, 255, 130)               # semi-transparent white


# initialize pygame
WINDOW_SIZE = 600
SQUARE_SIZE = WINDOW_SIZE // 8
PIECE_IMAGES = {}
pygame.init()
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Chess")

# loads piece images
def load_pieces():
    pieces = ['p', 'n', 'b', 'r', 'q', 'k']
    for piece in pieces:
        PIECE_IMAGES[f'w{piece}'] = pygame.transform.scale(
            pygame.image.load(os.path.join('assets', f'w{piece}.png')),
            (SQUARE_SIZE, SQUARE_SIZE)
        )
        PIECE_IMAGES[f'b{piece}'] = pygame.transform.scale(
            pygame.image.load(os.path.join('assets', f'b{piece}.png')),
            (SQUARE_SIZE, SQUARE_SIZE)
        )

# converts mouse position to board square
def get_square_from_mouse(pos, flip=False):
    x, y = pos
    if flip:
        x = WINDOW_SIZE - x
        y = WINDOW_SIZE - y
    return chess.Square(x // SQUARE_SIZE + 8 * (7 - y // SQUARE_SIZE))

# converts board square to screen coordinates
def get_screen_coords(square, flip=False):
    file_idx = chess.square_file(square)
    rank_idx = 7 - chess.square_rank(square)
    if flip:
        file_idx = 7 - file_idx
        rank_idx = 7 - rank_idx
    return (file_idx * SQUARE_SIZE, rank_idx * SQUARE_SIZE)

# draws the chess board checker pattern
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

# draw the chess pieces on the board
def draw_pieces(board, flip=False):
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = 'w' if piece.color else 'b'
            piece_img = PIECE_IMAGES[f'{color}{piece.symbol().lower()}']
            x, y = get_screen_coords(square, flip)
            screen.blit(piece_img, (x, y))

# draw highlights for selected piece, valid moves, and check
def draw_highlights(selected_square, valid_moves, board, last_move=None, flip=False):
    # the selected piece
    if selected_square is not None:
        x, y = get_screen_coords(selected_square, flip)
        highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
        pygame.draw.rect(highlight_surf, HIGHLIGHT, highlight_surf.get_rect())
        screen.blit(highlight_surf, (x, y))

    # the move indicators
    for move in valid_moves:
        x, y = get_screen_coords(move.to_square, flip)
        center_x = x + SQUARE_SIZE // 2
        center_y = y + SQUARE_SIZE // 2
        
        if board.piece_at(move.to_square):
            # for captures, draw a larger hollow circle
            radius = SQUARE_SIZE // 2.25
            circle_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, CAPTURE_HIGHLIGHT, 
                             (SQUARE_SIZE//2, SQUARE_SIZE//2), radius, 3)
            screen.blit(circle_surf, (x, y))
        else:
            # for regular moves, draw a small filled circle
            radius = SQUARE_SIZE // 4
            circle_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(circle_surf, MOVE_DOT,
                             (SQUARE_SIZE//2, SQUARE_SIZE//2), radius)
            screen.blit(circle_surf, (x, y))

    # the king in check
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

    # add last move highlight
    if last_move:
        for square in [last_move.from_square, last_move.to_square]:
            x, y = get_screen_coords(square, flip)
            highlight_surf = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
            pygame.draw.rect(highlight_surf, LAST_MOVE_HIGHLIGHT, highlight_surf.get_rect())
            screen.blit(highlight_surf, (x, y))

# show game over screen with appropriate message
def show_game_over_screen(board):
    overlay = pygame.Surface((WINDOW_SIZE, WINDOW_SIZE), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 100))
    screen.blit(overlay, (0, 0))

    font = pygame.font.Font('assets/caviar_dreams.ttf', 64)
    if board.is_checkmate():
        text = "Checkmate! {} wins!".format("Black" if board.turn == chess.WHITE else "White")
    elif board.is_stalemate():
        text = "Stalemate!"
    else:
        text = "Draw!"

    text_surface = font.render(text, True, (255, 255, 255))
    text_rect = text_surface.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 - 50))
    screen.blit(text_surface, text_rect)

    font = pygame.font.Font('assets/caviar_dreams.ttf', 32)
    restart_text = "Press SPACE to play again"
    restart_surface = font.render(restart_text, True, (255, 255, 255))
    restart_rect = restart_surface.get_rect(center=(WINDOW_SIZE/2, WINDOW_SIZE/2 + 50))
    screen.blit(restart_surface, restart_rect)
    pygame.display.flip()

# shows interface for pawn promotion
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
        
        piece_img = PIECE_IMAGES[f'{"w" if player_color else "b"}{symbol}']  # use player_color here
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

# choose player color: white, black, or random
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
                
def handle_events_during_bot_think():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        pygame.display.flip()

def main():
    load_pieces()
    board = chess.Board()
    selected_square = None
    valid_moves = []
    game_over = False
    last_move = None
    bot = ChessBot()

    player_color = choose_color()
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and game_over:
                if event.key == pygame.K_SPACE:
                    board = chess.Board()
                    selected_square = None
                    valid_moves = []
                    game_over = False
                    player_color = choose_color()  # prompt to choose color again
            elif event.type == pygame.MOUSEBUTTONDOWN and not game_over:
                if event.button == 1 and board.turn == player_color:  # left click and player's turn
                    flip = player_color == chess.BLACK
                    clicked_square = get_square_from_mouse(event.pos, flip)
                    
                    if selected_square is None:
                        piece = board.piece_at(clicked_square)
                        if piece and piece.color == board.turn:
                            selected_square = clicked_square
                            valid_moves = [move for move in board.legal_moves 
                                         if move.from_square == clicked_square]
                    else:
                        # create basic move
                        move = chess.Move(selected_square, clicked_square)
                        # check if this move exists in valid moves (either as is or as a promotion)
                        matching_moves = [m for m in valid_moves 
                                        if m.from_square == selected_square 
                                        and m.to_square == clicked_square]
                        
                        if matching_moves:
                            # if it's a pawn reaching the back rank
                            piece = board.piece_at(selected_square)
                            is_promotion = (piece 
                                          and piece.piece_type == chess.PAWN 
                                          and ((clicked_square >= chess.A8 and piece.color) 
                                               or (clicked_square <= chess.H1 and not piece.color)))
                            
                            if is_promotion:
                                promotion_piece = show_promotion_selection(flip, board.turn)
                                move = chess.Move(selected_square, clicked_square, promotion_piece)
                            else:
                                move = matching_moves[0]  # use the first matching move
                                
                            if move in board.legal_moves:
                                board.push(move)
                                last_move = move
                                
                        selected_square = None
                        valid_moves = []
                        
        if board.is_game_over() and not game_over:
            game_over = True
            bot.reset()

        if not game_over and board.turn != player_color:
            timer = {
                'start': time.time(),
                'time': 10
            }
            
            # Show "Thinking..." message
            font = pygame.font.Font('assets/caviar_dreams.ttf', 32)
            thinking_text = font.render("Thinking...", True, (255, 255, 255))
            thinking_rect = thinking_text.get_rect(center=(WINDOW_SIZE/2, 50))
            
            # Create semi-transparent overlay
            overlay = pygame.Surface((WINDOW_SIZE, 100))
            overlay.fill((0, 0, 0))
            overlay.set_alpha(150)
            
            # Draw current state with thinking message
            flip = player_color == chess.BLACK
            draw_board(flip)
            draw_highlights(selected_square, valid_moves, board, last_move, flip)
            draw_pieces(board, flip)
            screen.blit(overlay, (0, 0))
            screen.blit(thinking_text, thinking_rect)
            pygame.display.flip()
            
            # Modify the bot's think method to handle events
            def thinking_callback():
                handle_events_during_bot_think()
            
            # Add callback to bot instance
            bot.thinking_callback = thinking_callback
            
            move = bot.think(board, timer)
            board.push(move)
            last_move = move

        # draw everything
        flip = player_color == chess.BLACK
        draw_board(flip)
        draw_highlights(selected_square, valid_moves, board, last_move, flip)
        draw_pieces(board, flip)

        if board.is_game_over() and not game_over:
            game_over = True
            show_game_over_screen(board)
        elif not game_over:
            pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    main()
