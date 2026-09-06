use chess_again::core::Game;
use rand::prelude::IndexedRandom;

fn main() {
    let mut game =
        Game::starting_at_fen("7k/8/1p6/8/8/P7/8/7K b - - 0 1").expect("starting fen is legal.");
    let mut rng = rand::rng();

    // play a random game to completion
    while !game.legal_moves().is_empty() {
        let mv = *game
            .legal_moves()
            .choose(&mut rng)
            .expect("legal moves should not be empty.");
        game.make_move(mv).expect("move should be legal.");
    }
    println!("Game:\n{}", game.to_pgn().expect("game is terminated."));
}
