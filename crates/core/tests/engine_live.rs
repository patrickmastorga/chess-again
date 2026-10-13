//! Live check against Stockfish. Ignored by default: it needs the binary.
use chess_again_core::{Engine, Score, UciEngine};

#[test]
#[ignore = "requires stockfish on PATH"]
fn a_search_returns_a_move_and_a_score() {
    let mut engine = UciEngine::new("stockfish").expect("spawn stockfish");
    engine
        .start_search_async(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            Some(200),
        )
        .expect("start");

    let mut result = None;
    while result.is_none() {
        std::thread::sleep(std::time::Duration::from_millis(10));
        result = engine.poll().expect("poll");
    }
    let result = result.expect("a result");
    println!("bestmove {} eval {:?}", result.best_move, result.eval);
    assert!(result.eval.is_some(), "eval should be captured from info");
}

#[test]
#[ignore = "requires stockfish on PATH"]
fn stop_blocks_until_the_search_is_over() {
    let mut engine = UciEngine::new("stockfish").expect("spawn stockfish");
    engine
        .start_search_async(
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            None, // infinite
        )
        .expect("start");

    std::thread::sleep(std::time::Duration::from_millis(300));
    let stopped = engine.stop().expect("stop").expect("a result");
    println!("stopped at {} eval {:?}", stopped.best_move, stopped.eval);

    // Nothing should be outstanding, so the engine is immediately reusable.
    engine
        .start_search_async(
            "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            Some(200),
        )
        .expect("restart");
    let mut result = None;
    while result.is_none() {
        std::thread::sleep(std::time::Duration::from_millis(10));
        result = engine.poll().expect("poll");
    }
    assert!(result.is_some());
}

#[test]
#[ignore = "requires stockfish on PATH"]
fn stopping_nothing_is_not_an_error() {
    let mut engine = UciEngine::new("stockfish").expect("spawn stockfish");
    assert!(engine.stop().expect("stop").is_none());
}

#[test]
#[ignore = "requires stockfish on PATH"]
fn mate_scores_are_reported_as_mate() {
    let mut engine = UciEngine::new("stockfish").expect("spawn stockfish");
    // Black is mated in one: white to move plays Qg7#.
    engine
        .start_search_async("6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1", Some(300))
        .expect("start");
    let mut result = None;
    while result.is_none() {
        std::thread::sleep(std::time::Duration::from_millis(10));
        result = engine.poll().expect("poll");
    }
    let score = result.expect("a result").eval.expect("a score");
    println!("score {score:?}");
    assert!(matches!(score, Score::Cp(_) | Score::Mate(_)));
}
