import chess
import numpy as np
import time
import chess.polyglot
import chess.syzygy
from typing import List, Optional, Dict
from square_tables import create_square_tables

class ChessBot:
    def __init__(self):
        self.done = False
        self.transposition_table = {}
        self.history_table = np.zeros((7, 64), dtype=np.int32)
        self.square_tables = create_square_tables()
        # Pre-calculate piece values
        self.mg_values = np.array([0, 82, 337, 365, 477, 1025, 0], dtype=np.int16)
        self.eg_values = np.array([0, 94, 281, 297, 512, 936, 0], dtype=np.int16)
        self.piece_to_index = {
            None: 0,
            chess.PAWN: 1,
            chess.KNIGHT: 2,
            chess.BISHOP: 3,
            chess.ROOK: 4,
            chess.QUEEN: 5,
            chess.KING: 6,
        }
        self.opening_book = chess.polyglot.open_reader("book.bin")
        self.thinking_callback = None
           
        # Cache commonly used values
        self._piece_cache = {}
        
        # Cache developed pieces
        self.developed_pieces = {
            chess.WHITE: {
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
            },
            chess.BLACK: {
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
            }
        }
        
        self.already_moved_pieces = { }

    def eval(self, board: chess.Board) -> int:
        # Cache the board position
        key = board.fen()
        if key in self._piece_cache:
            return self._piece_cache[key]

        mg_eval = 0
        eg_eval = 0
        piece_count = 0

        # Loop through all pieces
        for piece_type in range(1, 7):
            w_pieces = list(board.pieces(piece_type, chess.WHITE))
            b_pieces = list(board.pieces(piece_type, chess.BLACK))
            piece_count += len(w_pieces) + len(b_pieces)
            
            # Start the evaluation
            # Get the piece values and square tables for white pieces
            if w_pieces:
                squares = np.array(w_pieces)
                ranks = squares // 8
                files = squares % 8
                mg_eval += (self.mg_values[piece_type] + 
                          self.square_tables[f'mg_{chess.PIECE_NAMES[piece_type]}'][ranks, files]).sum()
                eg_eval += (self.eg_values[piece_type] + 
                          self.square_tables[f'eg_{chess.PIECE_NAMES[piece_type]}'][ranks, files]).sum()

            # Get the piece values and square tables for black pieces
            if b_pieces:
                squares = np.array(b_pieces)
                ranks = 7 - (squares // 8)
                files = squares % 8
                mg_eval -= (self.mg_values[piece_type] + 
                          self.square_tables[f'mg_{chess.PIECE_NAMES[piece_type]}'][ranks, files]).sum()
                eg_eval -= (self.eg_values[piece_type] + 
                          self.square_tables[f'eg_{chess.PIECE_NAMES[piece_type]}'][ranks, files]).sum()

        eval_score = (mg_eval * piece_count + eg_eval * (32 - piece_count)) >> 5
        eval_score += self.punish_piece_positions(board)
        final_score = 25 + (eval_score if board.turn == chess.WHITE else -eval_score)
        
        # Cache the result
        self._piece_cache[key] = final_score
        return final_score
    
    def punish_piece_positions(self, board):
        punishment = 0
        # Check for doubled pawns and add 20 for each doubled white pawn and subtract 20 for each doubled black pawn
        for file in range(8):
            white_pawns = 0
            black_pawns = 0
            for rank in range(8):
                if board.piece_at(chess.square(file, rank)) == chess.PAWN:
                    if board.color_at(chess.square(file, rank)) == chess.WHITE:
                        white_pawns += 1
                    else:
                        black_pawns += 1
            if white_pawns > 1:
                punishment += 5
            if black_pawns > 1:
                punishment -= 5
                
        # Punish not developing knights and bishops after 7 moves
        if board.fullmove_number > 7:
            for color in [chess.WHITE, chess.BLACK]:
                punishing_factor = board.fullmove_number - 7
                if self.developed_pieces[color][chess.KNIGHT]:
                    print("Punishing undeveloped knight")
                    print(self.developed_pieces)
                    punishment += 10 * punishing_factor if color == chess.WHITE else -10 * punishing_factor
                if self.developed_pieces[color][chess.BISHOP]:
                    print("Punishing undeveloped bishop")
                    print(self.developed_pieces)
                    punishment += 10  * punishing_factor if color == chess.WHITE else -10 * punishing_factor
                    
        # Punish moving same piece multiple times during the opening
        if board.fullmove_number < 10:
            for move in board.move_stack:
                piece_type = board.piece_type_at(move.from_square)
                if move.from_square in self.already_moved_pieces:
                    if piece_type in self.already_moved_pieces[move.from_square]:
                        punishment -= 5 if board.turn == chess.WHITE else 5
                    else:
                        self.already_moved_pieces[move.from_square].add(piece_type)
                else:
                    self.already_moved_pieces[move.from_square] = { piece_type }
                
        return punishment

    def sort_moves(self, board: chess.Board, moves: List[chess.Move], tt_move: Optional[chess.Move]) -> List[chess.Move]:
        # Sort moves based on the history table and transposition table
        move_scores = np.zeros(len(moves), dtype=np.int32)
        
        for i, move in enumerate(moves):
            score = 0
            if move == tt_move:
                score = 1100000000
            elif move.promotion == chess.QUEEN:
                score = 1000000000
            elif board.is_capture(move):
                captured_piece = board.piece_type_at(move.to_square)
                moved_piece = board.piece_type_at(move.from_square)
                if self.piece_to_index[captured_piece] > self.piece_to_index[moved_piece]:
                    score = 990000000
                elif self.piece_to_index[captured_piece] == self.piece_to_index[moved_piece]:
                    score = 980000000
            score += self.history_table[board.piece_type_at(move.from_square), move.to_square]
            move_scores[i] = score

        return [moves[i] for i in np.argsort(-move_scores)]

    def alpha_beta(self, board: chess.Board, depth: int, timer: Dict, alpha: int, beta: int) -> int:
        # Callback for thinking
        if self.thinking_callback:
            self.thinking_callback()

        # Early exit conditions
        if not any(board.legal_moves):
            if board.is_checkmate():
                return -200000 + board.ply()
            return 0 # Stalemate condition?

        if time.time() - timer['start'] > timer['time']:
            self.done = True
            return 0

        # Transposition table lookup
        zobrist = chess.polyglot.zobrist_hash(board)
        if zobrist in self.transposition_table:
            entry = self.transposition_table[zobrist]
            if entry[1] >= depth:
                if entry[3] == 1:
                    return min(entry[2], beta)
                if entry[3] == 3 and entry[2] >= beta:
                    return beta
                if entry[3] == 2 and entry[2] <= alpha:
                    return alpha

        if depth == 0:
            eval_score = self.eval(board)
            self.transposition_table[zobrist] = (zobrist, depth, eval_score, 1, None)
            return eval_score

        moves = list(board.legal_moves)
        if not moves: # Double-check for no legal moves
            return 0
        
        best_move = moves[0]
        tt_move = self.transposition_table.get(zobrist, (None, None, None, None, None))[4]
        moves = self.sort_moves(board, moves, tt_move)

        # Principal Variation Search
        for move in moves:
            board.push(move)
            eval_score = -self.alpha_beta(board, depth - 1, timer, -beta, -alpha)
            board.pop()
            
            if self.done:
                return 0

            if eval_score >= beta:
                self.transposition_table[zobrist] = (zobrist, depth, beta, 3, move)
                if not board.is_capture(move):
                    self.history_table[board.piece_type_at(move.from_square), move.to_square] += depth * depth
                return beta
                
            if eval_score > alpha:
                alpha = eval_score
                best_move = move

        self.transposition_table[zobrist] = (zobrist, depth, alpha, 1 if alpha > alpha else 2, best_move)
        return alpha
    
    def think(self, board: chess.Board, timer: Dict) -> chess.Move:
        # Check if last move was knight or bishop move and update the developed pieces
        if board.move_stack:
            last_move = board.peek()
            if board.piece_type_at(last_move.from_square) in [chess.KNIGHT, chess.BISHOP]:
                self.developed_pieces[board.turn][board.piece_type_at(last_move.from_square)] = True
        
        try:
            move = self.opening_book.find(board).move
            print("Opening move found")
            if move:
                self.update_piece_moves(board, move)
            return move
        except:
            pass
        
        self.done = False
        moves = list(board.legal_moves)
        max_depth = 2
        final_move = moves[0]

        while not self.done:
            max_depth += 1
            moves = self.sort_moves(board, moves, final_move)
            best_eval = -1000000

            for move in moves:
                board.push(move)
                eval = -self.alpha_beta(board, max_depth, timer, -1000000, -best_eval)
                    
                board.pop()
                if self.done:
                    break
                if eval > best_eval:
                    final_move = move
                    best_eval = eval

        if final_move is None:
            return moves[0]
        
        # Update the piece moves
        self.update_piece_moves(board, final_move)
        return final_move
    
    def update_piece_moves(self, board: chess.Board, move: chess.Move):
        # If the move is a knight or bishop move, update the developed pieces
        if board.piece_type_at(move.from_square) in [chess.KNIGHT, chess.BISHOP]:
            self.developed_pieces[board.turn][board.piece_type_at(move.from_square)] += 1
        
            
        # Update the already moved pieces
        piece_type = board.piece_type_at(move.from_square)
        if move.from_square in self.already_moved_pieces:
            if piece_type in self.already_moved_pieces[move.from_square]:
                self.already_moved_pieces[move.from_square].remove(piece_type)
    
    def reset(self):
        print("Resetting bot")
        if self.opening_book:
            self.opening_book.close()
        self.done = False
        self.transposition_table = {}
        self.history_table = np.zeros((7, 64), dtype=np.int32)
        self._piece_cache = {}
        self.opening_book = chess.polyglot.open_reader("book.bin")
        self.developed_pieces = {
            chess.WHITE: {
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
            },
            chess.BLACK: {
                chess.KNIGHT: 0,
                chess.BISHOP: 0,
            }
        }