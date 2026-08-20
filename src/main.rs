use chess_again::core::BITBOARDS;
use chess_again::core::Board;

fn main() {
    let board = Board::new();
    println!("{}", board);

    println!("{}", BITBOARDS.white_pawn_attacks[12]);

    println!("{}", BITBOARDS.northwest_ray[12]);
}
