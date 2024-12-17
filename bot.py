import chess
import chess.polyglot
import numpy as np
from evaluation import Evaluation


class ChessBot:
    def __init__(self):
        self.eval = Evaluation()
        self.transposition_table = {}
        self.opening_book = chess.polyglot.open_reader("book.bin")
        self.positions_checked = 0
        self.thinking = False
    
    def get_best_move(self, fen):
        self.thinking = True
        # Try to get a move from the opening book
        try:
            entry = self.opening_book.find(chess.Board(fen))
            print(entry.move)
            return entry.move
        except Exception as e:
            print(e)
            pass
        board = chess.Board(fen)
        best_move = None
        best_eval = float('-inf')
        for move in self.get_legal_moves(board):
            self.positions_checked += 1
            print(self.positions_checked)
            board.push(move)
            eval = self.minimax(board, 3, float('-inf'), float('inf'), False)
            board.pop()
            if eval > best_eval:
                best_eval = eval
                best_move = move
                
        print("Positions checked: ", self.positions_checked)
        self.positions_checked = 0
        self.thinking = False
        return best_move
    
    def minimax(self, board, depth, alpha, beta, is_maximizing):
        # print("Depth: ", depth)
        if depth == 0:
            try:
                return self.transposition_table[board.fen()]
            except KeyError:
                eval = self.search_till_no_capture(board)
                return eval
        
        if is_maximizing:
            max_eval = float('-inf')
            for move in board.legal_moves:
                self.positions_checked += 1
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break
            return max_eval
        else:
            min_eval = float('inf')
            for move in board.legal_moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break
            return min_eval
        
    def search_till_no_capture(self, board):
        # Search till no capture is possible and then evaluate the board
        for move in board.legal_moves:
            if board.is_capture(move):
                board.push(move)
                eval = self.search_till_no_capture(board)
                board.pop()
                return eval
        
        evaluation = self.eval.evaluate(self.fen_to_bitboard(board.fen()))
        self.transposition_table[board.fen()] = evaluation
        return evaluation
        
    def get_legal_moves(self, board):
        # Get the legal moves for the current board and order them
        legal_moves = list(board.legal_moves)
        
        # We move capture, check and promotion moves to the front
        capture_moves = []
        check_moves = []
        promotion_moves = []
        normal_moves = []
        for move in legal_moves:
            if board.is_capture(move):
                capture_moves.append(move)
            elif board.is_check():
                check_moves.append(move)
            elif self.is_promotion(move, board):
                promotion_moves.append(move)
            else:
                normal_moves.append(move)
                
        return capture_moves + check_moves + promotion_moves + normal_moves
    
    def is_promotion(self, move, board):
        # Check if pawn is white and on the 8th rank or black and on the 1st rank
        if board.turn == chess.WHITE and chess.square_rank(move.to_square) == 7 and board.piece_at(move.from_square) == chess.Piece(chess.PAWN, chess.WHITE):
            return True
        elif board.turn == chess.BLACK and chess.square_rank(move.to_square) == 0 and board.piece_at(move.from_square) == chess.Piece(chess.PAWN, chess.BLACK):
            return True
        
        
    def fen_to_bitboard(self, fen):
        piece_layer = {
            'P': 0,
            'N': 1,
            'B': 2,
            'R': 3,
            'Q': 4,
            'K': 5,
            'p': 6,
            'n': 7,
            'b': 8,
            'r': 9,
            'q': 10,
            'k': 11
        }
    
        fen = fen.split(' ')
        bit_board = np.zeros((12, 8, 8))
        pieces = fen[0]
        rows = pieces.split('/')
        for i, row in enumerate(rows):
            j = 0
            for c in row:
                if c.isdigit():
                    j += int(c)
                else:
                    bit_board[piece_layer[c], i, j] = 1
                    j += 1
        
        return bit_board
    

if __name__ == '__main__':
    board = chess.Board()

    bot = ChessBot(chess.WHITE)
    bit_board = bot.fen_to_bitboard(board.fen())

    evaluation = Evaluation()
    print(evaluation.evaluate(bit_board))