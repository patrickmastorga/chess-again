use crate::core::{Bitboard, Board, Move, MoveType, PieceType};

pub trait MoveSink {
    fn push(&mut self, mv: Move);
}

impl MoveSink for Vec<Move> {
    fn push(&mut self, mv: Move) {
        Vec::push(self, mv);
    }
}

#[derive(Debug)]
struct Pins {
    pub any: Bitboard,
    northwest: Bitboard,
    northeast: Bitboard,
    southwest: Bitboard,
    southeast: Bitboard,
    north: Bitboard,
    south: Bitboard,
    east: Bitboard,
    west: Bitboard,
}

impl Pins {
    fn restrict_moves(&self, source: usize, destinations: Bitboard) -> Bitboard {
        if !self.any.contains(source) {
            return destinations;
        }
        if self.northwest.contains(source) {
            return destinations & self.northwest;
        }
        if self.northeast.contains(source) {
            return destinations & self.northeast;
        }
        if self.southwest.contains(source) {
            return destinations & self.southwest;
        }
        if self.southeast.contains(source) {
            return destinations & self.southeast;
        }
        if self.north.contains(source) {
            return destinations & self.north;
        }
        if self.south.contains(source) {
            return destinations & self.south;
        }
        if self.east.contains(source) {
            return destinations & self.east;
        }
        if self.west.contains(source) {
            return destinations & self.west;
        }
        unreachable!()
    }
}

impl Board {
    pub fn legal_moves(&mut self) -> Vec<Move> {
        let mut legal_moves = Vec::new();
        self.generate_legal_moves(&mut legal_moves);
        legal_moves
    }

    pub fn is_legal_move(&mut self, mv: Move) -> bool {
        self.legal_moves().contains(&mv)
    }

    pub fn make_move(&mut self, mv: Move) -> Result<Option<PieceType>, String> {
        if !self.is_legal_move(mv) {
            return Err(format!("Illegal move: {mv}"));
        }
        Ok(self.make_move_unchecked(mv))
    }

    pub fn perft(&mut self, depth: u32) -> u64 {
        let mut move_stack = Vec::new();
        self._perft(depth, &mut move_stack)
    }

    fn _perft(&mut self, depth: u32, move_stack: &mut Vec<Move>) -> u64 {
        if depth == 0 {
            return 1;
        }

        let mut nodes = 0;
        let stack_start = move_stack.len();
        self.generate_legal_moves(move_stack);
        while move_stack.len() > stack_start {
            let mv = move_stack.pop().unwrap();
            self.make_move_unchecked(mv);
            nodes += self._perft(depth - 1, move_stack);
            self.unmake_move();
        }
        nodes
    }

    pub fn generate_legal_moves<S: MoveSink>(&self, moves: &mut S) {
        let (num_checks, blocks, pins) = self.calculate_checks_and_pins();

        // king moves
        let king_index = self.king_index(self.active_color());
        for destination in self.king_moves(king_index).iter_indices() {
            if !self.is_attacked_ignore_king(destination) {
                moves.push(Move {
                    from: king_index as u8,
                    to: destination as u8,
                    move_type: MoveType::Regular,
                });
            }
        }

        // double check restricts to only king moves
        if num_checks > 1 {
            return;
        }

        // if the king is in check, only moves which block the check or capture the checking piece are legal
        let legal_destinations = if num_checks != 0 {
            blocks
        } else {
            Bitboard::ALL
        };

        // castling moves
        if num_checks == 0 {
            let all_pieces = self.all_pieces();
            if self.castling_rights().kingside(self.active_color())
                && !all_pieces.contains(king_index + 1)
                && !all_pieces.contains(king_index + 2)
                && !self.is_attacked_ignore_king(king_index + 1)
                && !self.is_attacked_ignore_king(king_index + 2)
            {
                moves.push(Move {
                    from: king_index as u8,
                    to: (king_index + 2) as u8,
                    move_type: MoveType::Castling(true),
                });
            }
            if self.castling_rights().queenside(self.active_color())
                && !all_pieces.contains(king_index - 1)
                && !all_pieces.contains(king_index - 2)
                && !all_pieces.contains(king_index - 3)
                && !self.is_attacked_ignore_king(king_index - 1)
                && !self.is_attacked_ignore_king(king_index - 2)
            {
                moves.push(Move {
                    from: king_index as u8,
                    to: (king_index - 2) as u8,
                    move_type: MoveType::Castling(false),
                });
            }
        }

        // pawn moves
        for source in self.pawns(self.active_color()).iter_indices() {
            let destinations =
                pins.restrict_moves(source, self.pawn_moves(source) & legal_destinations);
            for destination in destinations.iter_indices() {
                const BACK_RANKS: Bitboard = Bitboard(0xFF000000000000FF);
                if BACK_RANKS.contains(destination) {
                    // promotions
                    moves.push(Move {
                        from: source as u8,
                        to: destination as u8,
                        move_type: MoveType::Promotion(PieceType::Knight),
                    });
                    moves.push(Move {
                        from: source as u8,
                        to: destination as u8,
                        move_type: MoveType::Promotion(PieceType::Bishop),
                    });
                    moves.push(Move {
                        from: source as u8,
                        to: destination as u8,
                        move_type: MoveType::Promotion(PieceType::Rook),
                    });
                    moves.push(Move {
                        from: source as u8,
                        to: destination as u8,
                        move_type: MoveType::Promotion(PieceType::Queen),
                    });
                } else {
                    // regular pawn move
                    moves.push(Move {
                        from: source as u8,
                        to: destination as u8,
                        move_type: MoveType::Regular,
                    });
                }
            }
        }

        // en passant moves
        if let Some(ep_square) = self.en_passant_square() {
            let ep_square = ep_square as usize;
            let sources = BITBOARDS.pawn_attacks[1 - self.active_color() as usize][ep_square]
                & self.pawns(self.active_color());
            for source in sources.iter_indices() {
                if self.en_passant_legal(source, ep_square, legal_destinations, &pins) {
                    let target = (source & 0b111000) | (ep_square & 0b111);
                    moves.push(Move {
                        from: source as u8,
                        to: ep_square as u8,
                        move_type: MoveType::EnPassant(target as u8),
                    });
                }
            }
        }

        // knight moves
        for source in self.knights(self.active_color()).iter_indices() {
            let destinations =
                pins.restrict_moves(source, self.knight_moves(source) & legal_destinations);
            for destination in destinations.iter_indices() {
                moves.push(Move {
                    from: source as u8,
                    to: destination as u8,
                    move_type: MoveType::Regular,
                });
            }
        }

        // bishop moves
        for source in self.bishops(self.active_color()).iter_indices() {
            let destinations =
                pins.restrict_moves(source, self.bishop_moves(source) & legal_destinations);
            for destination in destinations.iter_indices() {
                moves.push(Move {
                    from: source as u8,
                    to: destination as u8,
                    move_type: MoveType::Regular,
                });
            }
        }

        // rook moves
        for source in self.rooks(self.active_color()).iter_indices() {
            let destinations =
                pins.restrict_moves(source, self.rook_moves(source) & legal_destinations);
            for destination in destinations.iter_indices() {
                moves.push(Move {
                    from: source as u8,
                    to: destination as u8,
                    move_type: MoveType::Regular,
                });
            }
        }

        // queen moves
        for source in self.queens(self.active_color()).iter_indices() {
            let destinations =
                pins.restrict_moves(source, self.queen_moves(source) & legal_destinations);
            for destination in destinations.iter_indices() {
                moves.push(Move {
                    from: source as u8,
                    to: destination as u8,
                    move_type: MoveType::Regular,
                });
            }
        }
    }

    fn calculate_checks_and_pins(&self) -> (usize, Bitboard, Pins) {
        let opponent = self.active_color().opposite();
        let enemy_pawns = self.pawns(opponent);
        let enemy_knights = self.knights(opponent);
        let diagonal_attackers = self.bishops(opponent) | self.queens(opponent);
        let straight_attackers = self.rooks(opponent) | self.queens(opponent);
        let king_index = self.king_index(self.active_color());
        let all_pieces = self.all_pieces();
        let friendly_pieces = self.pieces_of_color(self.active_color());

        let mut num_checks: usize = 0;
        let mut blocks = Bitboard::EMPTY;
        let mut pins = Pins {
            any: Bitboard::EMPTY,
            northwest: Bitboard::EMPTY,
            northeast: Bitboard::EMPTY,
            southwest: Bitboard::EMPTY,
            southeast: Bitboard::EMPTY,
            north: Bitboard::EMPTY,
            south: Bitboard::EMPTY,
            east: Bitboard::EMPTY,
            west: Bitboard::EMPTY,
        };

        let northeast_attacks = northeast_sliding_attacks(king_index, all_pieces);
        if northeast_attacks & diagonal_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= northeast_attacks;
        } else if northeast_attacks & friendly_pieces != Bitboard::EMPTY {
            let northeast_through_attacks =
                northeast_sliding_attacks(king_index, all_pieces & !northeast_attacks);
            if northeast_through_attacks & diagonal_attackers != Bitboard::EMPTY {
                pins.any |= northeast_through_attacks;
                pins.northeast = northeast_through_attacks;
            }
        }
        let northwest_attacks = northwest_sliding_attacks(king_index, all_pieces);
        if northwest_attacks & diagonal_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= northwest_attacks;
        } else if northwest_attacks & friendly_pieces != Bitboard::EMPTY {
            let northwest_through_attacks =
                northwest_sliding_attacks(king_index, all_pieces & !northwest_attacks);
            if northwest_through_attacks & diagonal_attackers != Bitboard::EMPTY {
                pins.any |= northwest_through_attacks;
                pins.northwest = northwest_through_attacks;
            }
        }
        let southeast_attacks = southeast_sliding_attacks(king_index, all_pieces);
        if southeast_attacks & diagonal_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= southeast_attacks;
        } else if southeast_attacks & friendly_pieces != Bitboard::EMPTY {
            let southeast_through_attacks =
                southeast_sliding_attacks(king_index, all_pieces & !southeast_attacks);
            if southeast_through_attacks & diagonal_attackers != Bitboard::EMPTY {
                pins.any |= southeast_through_attacks;
                pins.southeast = southeast_through_attacks;
            }
        }
        let southwest_attacks = southwest_sliding_attacks(king_index, all_pieces);
        if southwest_attacks & diagonal_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= southwest_attacks;
        } else if southwest_attacks & friendly_pieces != Bitboard::EMPTY {
            let southwest_through_attacks =
                southwest_sliding_attacks(king_index, all_pieces & !southwest_attacks);
            if southwest_through_attacks & diagonal_attackers != Bitboard::EMPTY {
                pins.any |= southwest_through_attacks;
                pins.southwest = southwest_through_attacks;
            }
        }

        let north_attacks = north_sliding_attacks(king_index, all_pieces);
        if north_attacks & straight_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= north_attacks;
        } else if north_attacks & friendly_pieces != Bitboard::EMPTY {
            let north_through_attacks =
                north_sliding_attacks(king_index, all_pieces & !north_attacks);
            if north_through_attacks & straight_attackers != Bitboard::EMPTY {
                pins.any |= north_through_attacks;
                pins.north = north_through_attacks;
            }
        }
        let south_attacks = south_sliding_attacks(king_index, all_pieces);
        if south_attacks & straight_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= south_attacks;
        } else if south_attacks & friendly_pieces != Bitboard::EMPTY {
            let south_through_attacks =
                south_sliding_attacks(king_index, all_pieces & !south_attacks);
            if south_through_attacks & straight_attackers != Bitboard::EMPTY {
                pins.any |= south_through_attacks;
                pins.south = south_through_attacks;
            }
        }
        let east_attacks = east_sliding_attacks(king_index, all_pieces);
        if east_attacks & straight_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= east_attacks;
        } else if east_attacks & friendly_pieces != Bitboard::EMPTY {
            let east_through_attacks = east_sliding_attacks(king_index, all_pieces & !east_attacks);
            if east_through_attacks & straight_attackers != Bitboard::EMPTY {
                pins.any |= east_through_attacks;
                pins.east = east_through_attacks;
            }
        }
        let west_attacks = west_sliding_attacks(king_index, all_pieces);
        if west_attacks & straight_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= west_attacks;
        } else if west_attacks & friendly_pieces != Bitboard::EMPTY {
            let west_through_attacks = west_sliding_attacks(king_index, all_pieces & !west_attacks);
            if west_through_attacks & straight_attackers != Bitboard::EMPTY {
                pins.any |= west_through_attacks;
                pins.west = west_through_attacks;
            }
        }

        let knight_attackers = BITBOARDS.knight_attacks[king_index] & enemy_knights;
        if knight_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= knight_attackers;
        }

        let pawn_attackers =
            BITBOARDS.pawn_attacks[self.active_color() as usize][king_index] & enemy_pawns;
        if pawn_attackers != Bitboard::EMPTY {
            num_checks += 1;
            blocks |= pawn_attackers;
        }
        (num_checks, blocks, pins)
    }

    fn is_attacked_ignore_king(&self, square: usize) -> bool {
        let opponent = self.active_color().opposite();
        let enemy_pawns = self.pawns(opponent);
        let enemy_knights = self.knights(opponent);
        let enemy_kings = self.kings(opponent);
        let diagonal_attackers = self.bishops(opponent) | self.queens(opponent);
        let straight_attackers = self.rooks(opponent) | self.queens(opponent);
        let occupancy = self.all_pieces() & !self.kings(self.active_color());

        !(BITBOARDS.pawn_attacks[self.active_color() as usize][square] & enemy_pawns).is_empty()
            || !(BITBOARDS.knight_attacks[square] & enemy_knights).is_empty()
            || !(BITBOARDS.king_attacks[square] & enemy_kings).is_empty()
            || !(bishop_attacks(square, occupancy) & diagonal_attackers).is_empty()
            || !(rook_attacks(square, occupancy) & straight_attackers).is_empty()
    }

    fn pawn_moves(&self, square: usize) -> Bitboard {
        let all_pieces = self.all_pieces();
        let mut pawn_sliding_moves = BITBOARDS.pawn_moves[self.active_color() as usize][square];
        pawn_sliding_moves &= !all_pieces;
        if !pawn_sliding_moves.is_empty() {
            let pawn_double_move =
                BITBOARDS.pawn_double_moves[self.active_color() as usize][square];
            pawn_sliding_moves |= pawn_double_move & !all_pieces;
        }

        let opponent = self.active_color().opposite();
        let enemy_pieces = self.pieces_of_color(opponent);
        let pawn_attacks = BITBOARDS.pawn_attacks[self.active_color() as usize][square];
        pawn_sliding_moves | (pawn_attacks & enemy_pieces)
    }

    fn knight_moves(&self, square: usize) -> Bitboard {
        let friendly_pieces = self.pieces_of_color(self.active_color());
        let knight_attacks = BITBOARDS.knight_attacks[square];
        knight_attacks & !friendly_pieces
    }

    fn bishop_moves(&self, square: usize) -> Bitboard {
        let friendly_pieces = self.pieces_of_color(self.active_color());
        let all_pieces = self.all_pieces();
        let bishop_attacks = bishop_attacks(square, all_pieces);
        bishop_attacks & !friendly_pieces
    }

    fn rook_moves(&self, square: usize) -> Bitboard {
        let friendly_pieces = self.pieces_of_color(self.active_color());
        let all_pieces = self.all_pieces();
        let rook_attacks = rook_attacks(square, all_pieces);
        rook_attacks & !friendly_pieces
    }

    fn queen_moves(&self, square: usize) -> Bitboard {
        let friendly_pieces = self.pieces_of_color(self.active_color());
        let all_pieces = self.all_pieces();
        let queen_attacks = rook_attacks(square, all_pieces) | bishop_attacks(square, all_pieces);
        queen_attacks & !friendly_pieces
    }

    fn king_moves(&self, square: usize) -> Bitboard {
        let friendly_pieces = self.pieces_of_color(self.active_color());
        BITBOARDS.king_attacks[square] & !friendly_pieces
    }

    fn en_passant_legal(
        &self,
        source: usize,
        ep_square: usize,
        legal_destinations: Bitboard,
        pins: &Pins,
    ) -> bool {
        let king_index = self.king_index(self.active_color());
        let target = (source & 0b111000) | (ep_square & 0b111);
        if (!legal_destinations.contains(ep_square) && !legal_destinations.contains(target))
            || pins
                .restrict_moves(source, Bitboard::from_square(ep_square))
                .is_empty()
        {
            return false;
        }
        let opponent = self.active_color().opposite();
        let diagonal_attackers = self.bishops(opponent) | self.queens(opponent);
        let straight_attackers = self.rooks(opponent) | self.queens(opponent);
        let occupancy = self
            .all_pieces()
            .with_set(ep_square)
            .with_clear(source)
            .with_clear(target);
        (bishop_attacks(king_index, occupancy) & diagonal_attackers).is_empty()
            && (rook_attacks(king_index, occupancy) & straight_attackers).is_empty()
    }
}

fn northeast_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.northeast_ray[square];
    match (ray & occupancy).lsb_index() {
        Some(blocking_index) => ray & !BITBOARDS.northeast_ray[blocking_index],
        None => ray,
    }
}

fn northwest_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.northwest_ray[square];
    match (ray & occupancy).lsb_index() {
        Some(blocking_index) => ray & !BITBOARDS.northwest_ray[blocking_index],
        None => ray,
    }
}

fn southeast_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.southeast_ray[square];
    match (ray & occupancy).msb_index() {
        Some(blocking_index) => ray & !BITBOARDS.southeast_ray[blocking_index],
        None => ray,
    }
}

fn southwest_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.southwest_ray[square];
    match (ray & occupancy).msb_index() {
        Some(blocking_index) => ray & !BITBOARDS.southwest_ray[blocking_index],
        None => ray,
    }
}

fn north_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.north_ray[square];
    match (ray & occupancy).lsb_index() {
        Some(blocking_index) => ray & !BITBOARDS.north_ray[blocking_index],
        None => ray,
    }
}

fn south_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.south_ray[square];
    match (ray & occupancy).msb_index() {
        Some(blocking_index) => ray & !BITBOARDS.south_ray[blocking_index],
        None => ray,
    }
}

fn east_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.east_ray[square];
    match (ray & occupancy).lsb_index() {
        Some(blocking_index) => ray & !BITBOARDS.east_ray[blocking_index],
        None => ray,
    }
}

fn west_sliding_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    let ray = BITBOARDS.west_ray[square];
    match (ray & occupancy).msb_index() {
        Some(blocking_index) => ray & !BITBOARDS.west_ray[blocking_index],
        None => ray,
    }
}

fn bishop_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    northeast_sliding_attacks(square, occupancy)
        | northwest_sliding_attacks(square, occupancy)
        | southeast_sliding_attacks(square, occupancy)
        | southwest_sliding_attacks(square, occupancy)
}

fn rook_attacks(square: usize, occupancy: Bitboard) -> Bitboard {
    north_sliding_attacks(square, occupancy)
        | south_sliding_attacks(square, occupancy)
        | east_sliding_attacks(square, occupancy)
        | west_sliding_attacks(square, occupancy)
}

pub struct BitboardTables {
    pub north_ray: [Bitboard; 64],
    pub south_ray: [Bitboard; 64],
    pub east_ray: [Bitboard; 64],
    pub west_ray: [Bitboard; 64],
    pub northeast_ray: [Bitboard; 64],
    pub northwest_ray: [Bitboard; 64],
    pub southeast_ray: [Bitboard; 64],
    pub southwest_ray: [Bitboard; 64],
    pub knight_attacks: [Bitboard; 64],
    pub king_attacks: [Bitboard; 64],
    pub pawn_attacks: [[Bitboard; 64]; 2],
    pub pawn_moves: [[Bitboard; 64]; 2],
    pub pawn_double_moves: [[Bitboard; 64]; 2],
}

const fn generate_bitboard_tables() -> BitboardTables {
    let mut north_ray = [Bitboard::EMPTY; 64];
    let mut south_ray = [Bitboard::EMPTY; 64];
    let mut east_ray = [Bitboard::EMPTY; 64];
    let mut west_ray = [Bitboard::EMPTY; 64];
    let mut northeast_ray = [Bitboard::EMPTY; 64];
    let mut northwest_ray = [Bitboard::EMPTY; 64];
    let mut southeast_ray = [Bitboard::EMPTY; 64];
    let mut southwest_ray = [Bitboard::EMPTY; 64];
    let mut knight_attacks = [Bitboard::EMPTY; 64];
    let mut king_attacks = [Bitboard::EMPTY; 64];
    let mut pawn_attacks = [[Bitboard::EMPTY; 64]; 2];
    let mut pawn_moves = [[Bitboard::EMPTY; 64]; 2];
    let mut pawn_double_moves = [[Bitboard::EMPTY; 64]; 2];

    let mut square = 0;
    while square < 64 {
        let rank = square / 8;
        let file = square % 8;

        // Sliding rays.  The source square is deliberately not included.
        let mut distance = 1;
        while rank + distance < 8 {
            north_ray[square].set(square + distance * 8);
            distance += 1;
        }

        distance = 1;
        while distance <= rank {
            south_ray[square].set(square - distance * 8);
            distance += 1;
        }

        distance = 1;
        while file + distance < 8 {
            east_ray[square].set(square + distance);
            distance += 1;
        }

        distance = 1;
        while distance <= file {
            west_ray[square].set(square - distance);
            distance += 1;
        }

        distance = 1;
        while rank + distance < 8 && file + distance < 8 {
            northeast_ray[square].set(square + distance * 9);
            distance += 1;
        }

        distance = 1;
        while rank + distance < 8 && distance <= file {
            northwest_ray[square].set(square + distance * 7);
            distance += 1;
        }

        distance = 1;
        while distance <= rank && file + distance < 8 {
            southeast_ray[square].set(square - distance * 7);
            distance += 1;
        }

        distance = 1;
        while distance <= rank && distance <= file {
            southwest_ray[square].set(square - distance * 9);
            distance += 1;
        }

        // Non-sliding attacks.
        let knight_offsets = [
            (1isize, 2isize),
            (2, 1),
            (2, -1),
            (1, -2),
            (-1, -2),
            (-2, -1),
            (-2, 1),
            (-1, 2),
        ];
        let mut offset = 0;
        while offset < knight_offsets.len() {
            let target_file = file as isize + knight_offsets[offset].0;
            let target_rank = rank as isize + knight_offsets[offset].1;
            if target_file >= 0 && target_file < 8 && target_rank >= 0 && target_rank < 8 {
                knight_attacks[square].set((target_rank * 8 + target_file) as usize);
            }
            offset += 1;
        }

        let mut target_rank = rank.saturating_sub(1);
        while target_rank <= rank + 1 && target_rank < 8 {
            let mut target_file = file.saturating_sub(1);
            while target_file <= file + 1 && target_file < 8 {
                if target_rank != rank || target_file != file {
                    king_attacks[square].set(target_rank * 8 + target_file);
                }
                target_file += 1;
            }
            target_rank += 1;
        }

        // White pawns (index 0) move towards increasing ranks.
        if rank < 7 {
            pawn_moves[0][square].set(square + 8);
            if rank == 1 {
                pawn_double_moves[0][square].set(square + 16);
            }
            if file > 0 {
                pawn_attacks[0][square].set(square + 7);
            }
            if file < 7 {
                pawn_attacks[0][square].set(square + 9);
            }
        }

        // Black pawns (index 1) move towards decreasing ranks.
        if rank > 0 {
            pawn_moves[1][square].set(square - 8);
            if rank == 6 {
                pawn_double_moves[1][square].set(square - 16);
            }
            if file > 0 {
                pawn_attacks[1][square].set(square - 9);
            }
            if file < 7 {
                pawn_attacks[1][square].set(square - 7);
            }
        }

        square += 1;
    }

    BitboardTables {
        north_ray,
        south_ray,
        east_ray,
        west_ray,
        northeast_ray,
        northwest_ray,
        southeast_ray,
        southwest_ray,
        knight_attacks,
        king_attacks,
        pawn_attacks,
        pawn_moves,
        pawn_double_moves,
    }
}

pub static BITBOARDS: BitboardTables = generate_bitboard_tables();
