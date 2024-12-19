import chess
import numpy as np
import time
import chess.polyglot

class ChessBot:
	def __init__(self):
		self.done = False
		self.transposition_table_size = 32768
		self.transposition_table = [None] * self.transposition_table_size
		self.history_table = np.zeros((7, 64), dtype=int)
		self.square_tables = self.create_square_tables()
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
		self.can_use_opening_book = True
    
	def __del__(self):
		self.opening_book.close()

	def create_square_tables(self):
		tables = {
			'mg_pawn': np.array([
				[0, 0, 0, 0, 0, 0, 0, 0],
				[98, 134, 61, 95, 68, 126, 34, -11],
				[-6, 7, 26, 31, 65, 56, 25, -20],
				[-14, 13, 6, 21, 23, 12, 17, -23],
				[-27, -2, -5, 12, 17, 6, 10, -25],
				[-26, -4, -4, -10, 3, 3, 33, -12],
				[-35, -1, -20, -23, -15, 24, 38, -22],
				[0, 0, 0, 0, 0, 0, 0, 0]
			]),
			'mg_knight': np.array([
				[-167, -89, -34, -49, 61, -97, -15, -107],
				[-73, -41, 72, 36, 23, 62, 7, -17],
				[-47, 60, 37, 65, 84, 129, 73, 44],
				[-9, 17, 19, 53, 37, 69, 18, 22],
				[-13, 4, 16, 13, 28, 19, 21, -8],
				[-23, -9, 12, 10, 19, 17, 25, -16],
				[-29, -53, -12, -3, -1, 18, -14, -19],
				[-105, -21, -58, -33, -17, -28, -19, -23]
			]),
			'mg_bishop': np.array([
				[-29, 4, -82, -37, -25, -42, 7, -8],
				[-26, 16, -18, -13, 30, 59, 18, -47],
				[-16, 37, 43, 40, 35, 50, 37, -2],
				[-4, 5, 19, 50, 37, 37, 7, -2],
				[-6, 13, 13, 26, 34, 12, 10, 4],
				[0, 15, 15, 15, 14, 27, 18, 10],
				[4, 15, 16, 0, 7, 21, 33, 1],
				[-33, -3, -14, -21, -13, -12, -39, -21]
			]),
			'mg_rook': np.array([
				[32, 42, 32, 51, 63, 9, 31, 43],
				[27, 32, 58, 62, 80, 67, 26, 44],
				[-5, 19, 26, 36, 17, 45, 61, 16],
				[-24, -11, 7, 26, 24, 35, -8, -20],
				[-36, -26, -12, -1, 9, -7, 6, -23],
				[-45, -25, -16, -17, 3, 0, -5, -33],
				[-44, -16, -20, -9, -1, 11, -6, -71],
				[-19, -13, 1, 17, 16, 7, -37, -26]
			]),
			'mg_queen': np.array([
				[-28, 0, 29, 12, 59, 44, 43, 45],
				[-24, -39, -5, 1, -16, 57, 28, 54],
				[-13, -17, 7, 8, 29, 56, 47, 57],
				[-27, -27, -16, -16, -1, 17, -2, 1],
				[-9, -26, -9, -10, -2, -4, 3, -3],
				[-14, 2, -11, -2, -5, 2, 14, 5],
				[-35, -8, 11, 2, 8, 15, -3, 1],
				[-1, -18, -9, 10, -15, -25, -31, -50]
			]),
			'mg_king': np.array([
				[-65, 23, 16, -15, -56, -34, 2, 13],
				[29, -1, -20, -7, -8, -4, -38, -29],
				[-9, 24, 2, -16, -20, 6, 22, -22],
				[-17, -20, -12, -27, -30, -25, -14, -36],
				[-49, -1, -27, -39, -46, -44, -33, -51],
				[-14, -14, -22, -46, -44, -30, -15, -27],
				[1, 7, -8, -64, -43, -16, 9, 8],
				[-15, 36, 12, -54, 8, -28, 24, 14]
			]),
			'eg_pawn': np.array([
				[0, 0, 0, 0, 0, 0, 0, 0],
				[178, 173, 158, 134, 147, 132, 165, 187],
				[94, 100, 85, 67, 56, 53, 82, 84],
				[32, 24, 13, 5, -2, 4, 17, 17],
				[13, 9, -3, -7, -7, -8, 3, -1],
				[4, 7, -6, 1, 0, -5, -1, -8],
				[13, 8, 8, 10, 13, 0, 2, -7],
				[0, 0, 0, 0, 0, 0, 0, 0]
			]),
			'eg_knight': np.array([
				[-58, -38, -13, -28, -31, -27, -63, -99],
				[-25, -8, -25, -2, -9, -25, -24, -52],
				[-24, -20, 10, 9, -1, -9, -19, -41],
				[-17, 3, 22, 22, 22, 11, 8, -18],
				[-18, -6, 16, 25, 16, 17, 4, -18],
				[-23, -3, -1, 15, 10, -3, -20, -22],
				[-42, -20, -10, -5, -2, -20, -23, -44],
				[-29, -51, -23, -15, -22, -18, -50, -64]
			]),
			'eg_bishop': np.array([
				[-14, -21, -11, -8, -7, -9, -17, -24],
				[-8, -4, 7, -12, -3, -13, -4, -14],
				[2, -8, 0, -1, -2, 6, 0, 4],
				[-3, 9, 12, 9, 14, 10, 3, 2],
				[-6, 3, 13, 19, 7, 10, -3, -9],
				[-12, -3, 8, 10, 13, 3, -7, -15],
				[-14, -18, -7, -1, 4, -9, -15, -27],
				[-23, -9, -23, -5, -9, -16, -5, -17]
			]),
			'eg_rook': np.array([
				[13, 10, 18, 15, 12, 12, 8, 5],
				[11, 13, 13, 11, -3, 3, 8, 3],
				[7, 7, 7, 5, 4, -3, -5, -3],
				[4, 3, 13, 1, 2, 1, -1, 2],
				[3, 5, 8, 4, -5, -6, -8, -11],
				[-4, 0, -5, -1, -7, -12, -8, -16],
				[-6, -6, 0, 2, -9, -9, -11, -3],
				[-9, 2, 3, -1, -5, -13, 4, -20]
			]),
			'eg_queen': np.array([
				[-9, 22, 22, 27, 27, 19, 10, 20],
				[-17, 20, 32, 41, 58, 25, 30, 0],
				[-20, 6, 9, 49, 47, 35, 19, 9],
				[3, 22, 24, 45, 57, 40, 57, 36],
				[-18, 28, 19, 47, 31, 34, 39, 23],
				[-16, -27, 15, 6, 9, 17, 10, 5],
				[-22, -23, -30, -16, -16, -23, -36, -32],
				[-33, -28, -22, -43, -5, -32, -20, -41]
			]),
			'eg_king': np.array([
				[-74, -35, -18, -18, -11, 15, 4, -17],
				[-12, 17, 14, 17, 17, 38, 23, 11],
				[10, 17, 23, 15, 20, 45, 44, 13],
				[-8, 22, 24, 27, 26, 33, 26, 3],
				[-18, -4, 21, 24, 27, 23, 9, -11],
				[-19, -3, 11, 21, 23, 16, 7, -9],
				[-27, -11, 4, 13, 14, 4, -5, -17],
				[-53, -34, -21, -11, -28, -14, -24, -43]
			])
		}
		return tables

	def sort_moves(self, board, moves, tt_move):
		values = []
		for move in moves:
			value = 0
			if move == tt_move:
				value = 1100000000
			if move.promotion is not None and board.piece_type_at(move.to_square) == chess.QUEEN:
				value = 1000000000
			if board.is_capture(move):
				captured_piece = board.piece_type_at(move.to_square)
				moved_piece = board.piece_type_at(move.from_square)
				if self.piece_to_index[captured_piece] > self.piece_to_index[moved_piece]:
					value = 990000000
				elif self.piece_to_index[captured_piece] == self.piece_to_index[moved_piece]:
					value = 980000000
			value += self.history_table[board.piece_type_at(move.from_square), move.to_square]
			values.append(value)
		sorted_moves = [move for _, move in sorted(zip(values, moves), key=lambda x: x[0], reverse=True)]
		return sorted_moves

	def eval(self, board):
		mg_eval = 0
		eg_eval = 0
		piece_count = 0

		mg_value = [82, 337, 365, 477, 1025, 0]
		eg_value = [94, 281, 297, 512, 936, 0]

		for piece_type in range(1, 7):
			for square in board.pieces(piece_type, chess.WHITE):
				piece_count += 1
				mg_eval += mg_value[piece_type - 1] + self.square_tables[f'mg_{chess.PIECE_NAMES[piece_type]}'][square // 8][square % 8]
				eg_eval += eg_value[piece_type - 1] + self.square_tables[f'eg_{chess.PIECE_NAMES[piece_type]}'][square // 8][square % 8]
			for square in board.pieces(piece_type, chess.BLACK):
				piece_count += 1
				mg_eval -= mg_value[piece_type - 1] + self.square_tables[f'mg_{chess.PIECE_NAMES[piece_type]}'][7 - square // 8][square % 8]
				eg_eval -= eg_value[piece_type - 1] + self.square_tables[f'eg_{chess.PIECE_NAMES[piece_type]}'][7 - square // 8][square % 8]

		eval = (mg_eval * piece_count + eg_eval * (32 - piece_count)) // 32
		eval += punish_piece_positions(board)
		return 25 + (eval if board.turn == chess.WHITE else -eval)

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
				punishment += 20
			if black_pawns > 1:
				punishment -= 20

	def insert_zobrist(self, idx, zobrist, depth, eval, type, move):
		self.transposition_table[idx] = (zobrist, depth, eval, type, move)

	def alpha_beta(self, board, depth, timer, alpha, beta):
		if board.is_checkmate():
			return -200000 + board.ply()

		zobrist = chess.polyglot.zobrist_hash(board)
		idx = zobrist % self.transposition_table_size
		starting_alpha = alpha

		if time.time() - timer['start'] > timer['time']:
			self.done = True
			return 0

		table_hit = False
		if self.transposition_table[idx] is not None:
			zobrist_entry = self.transposition_table[idx]
			zobrist_zobrist, zobrist_depth, zobrist_eval, zobrist_type, zobrist_move = zobrist_entry
			if zobrist_zobrist == zobrist:
				table_hit = True
				if zobrist_depth >= depth:
					if zobrist_type == 1:
						return min(zobrist_eval, beta)
					if zobrist_type == 3 and zobrist_eval >= beta:
						return beta
					if zobrist_type == 2 and zobrist_eval <= alpha:
						return alpha

		if depth == 0:
			eval = self.eval(board)
			self.insert_zobrist(idx, zobrist, depth, eval, 1, None)
			return eval

		best_move = None
		zobrist_move = None
		if table_hit:
			zobrist_move = zobrist_entry[4]

		moves = list(board.legal_moves)
		moves = self.sort_moves(board, moves, zobrist_move)
		if best_move is None:
			best_move = moves[0]

		for move in moves:
			board.push(move)
			eval = -self.alpha_beta(board, depth - 1, timer, -beta, -alpha)
			board.pop()
			if self.done:
				return 0

			if eval >= beta:
				self.insert_zobrist(idx, zobrist, depth, beta, 3, move)
				if not board.is_capture(move):
					self.history_table[board.piece_type_at(move.from_square), move.to_square] += depth * depth
				return beta
			if eval > alpha:
				alpha = eval
				best_move = move

		self.insert_zobrist(idx, zobrist, depth, alpha, 1 if alpha > starting_alpha else 2, best_move)
		return alpha

	def think(self, board, timer):
		if self.can_use_opening_book:
			try:
				move = self.opening_book.find(board).move
				return move
			except:
				self.can_use_opening_book = False
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
				print(f"Move: {move} Eval: {eval}")
				board.pop()
				if self.done:
					break
				if eval > best_eval:
					final_move = move
					best_eval = eval

		return final_move
	
if __name__ == "__main__":
    bot = ChessBot()
    board = chess.Board()
    
    timer = {
        'start': time.time(),
        'time': 10
            }
    move = bot.think(board, timer)
    print(move)