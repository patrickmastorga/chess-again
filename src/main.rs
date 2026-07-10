use chess_again::core::Board;

fn main() {
    let board: Board = Board::new();
    println!("Chess engine board evaluation: {}", board.evaluate());
}