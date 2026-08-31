use chess_again::core::{Board, Move, MoveType, PieceType, utils};
use std::io::{self, Write};

fn main() {
    // 4k3/6b1/8/3Pp3/8/8/8/K7 w - e6 0 1
    let mut board = loop {
        let fen = read_input("Enter FEN: ");
        match Board::from_fen(&fen) {
            Ok(board) => break board,
            Err(error) => println!("{error}"),
        }
    };
    let depth = loop {
        let input = read_input("Enter perft depth: ");
        match input.parse::<u32>() {
            Ok(depth) => break depth,
            Err(_) => println!("Invalid depth: '{input}'"),
        }
    };

    for current_depth in (0..=depth).rev() {
        println!("Current board FEN: {}", board.to_fen());
        perft(&mut board, current_depth);

        if current_depth == 0 {
            break;
        }

        loop {
            let uci_move = read_input("Enter the move in UCI notation: ");

            match from_uci(&board, &uci_move) {
                Ok(parsed_move) => match board.make_move(parsed_move) {
                    Ok(_) => break,
                    Err(error) => println!("{error}"),
                },
                Err(error) => println!("{error}"),
            }
        }
    }
}

fn read_input(prompt: &str) -> String {
    print!("{prompt}");
    io::stdout().flush().expect("Failed to flush stdout.");

    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input.");
    input.trim().to_string()
}

fn perft(board: &mut Board, depth: u32) {
    if depth == 0 {
        return;
    }

    let mut nodes = 0;
    for mv in board.legal_moves().to_vec() {
        board.make_move(mv).expect("Move should be legal.");
        let current_nodes = board.perft(depth - 1);
        println!("{}: {}", mv, current_nodes);
        nodes += current_nodes;
        board.unmake_move();
    }
    println!("\nNodes searched: {}", nodes);
}

fn from_uci(board: &Board, input: &str) -> Result<Move, String> {
    let input = input.trim().to_ascii_lowercase();
    let input_bytes = input.as_bytes();
    if input_bytes.len() != 4 && input_bytes.len() != 5 {
        return Err(format!("Invalid UCI move: '{input}'"));
    }

    let from =
        utils::parse_algebraic(&input[0..2]).map_err(|_| format!("Invalid UCI move: '{input}'"))?;
    let to =
        utils::parse_algebraic(&input[2..4]).map_err(|_| format!("Invalid UCI move: '{input}'"))?;
    let move_type = match input_bytes.get(4) {
        None => MoveType::Regular,
        Some(b'n') => MoveType::Promotion(PieceType::Knight),
        Some(b'b') => MoveType::Promotion(PieceType::Bishop),
        Some(b'r') => MoveType::Promotion(PieceType::Rook),
        Some(b'q') => MoveType::Promotion(PieceType::Queen),
        Some(_) => return Err(format!("Invalid UCI promotion: '{input}'")),
    };

    let move_type = if matches!(move_type, MoveType::Regular)
        && board.piece_at(from).map(|(piece_type, _)| piece_type) == Some(PieceType::King)
        && from.abs_diff(to) == 2
    {
        MoveType::Castling(to > from)
    } else if matches!(move_type, MoveType::Regular)
        && board.piece_at(from).map(|(piece_type, _)| piece_type) == Some(PieceType::Pawn)
        && board.en_passant_square() == Some(to as u8)
        && from % 8 != to % 8
    {
        let target = (from & 0b111000) | (to & 0b111);
        MoveType::EnPassant(target as u8)
    } else {
        move_type
    };

    Ok(Move {
        from: from as u8,
        to: to as u8,
        move_type,
    })
}
