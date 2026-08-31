use chess_again::core::Board;

fn main() {
    let mut board =
        Board::from_fen("rnbqkb1r/ppppp1pp/7n/4Pp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3").unwrap();
    let nodes = board.perft(1);
    println!("{nodes} nodes");
}
