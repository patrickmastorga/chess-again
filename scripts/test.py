import chess_again

engine = chess_again.Engine()
print("Chess engine evaluation:", engine.evaluate())

fenloader = chess_again.FenLoader()
print("FENs loaded:", fenloader.get_fens())
