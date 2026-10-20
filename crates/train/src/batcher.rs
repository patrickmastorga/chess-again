use burn::data::dataloader::batcher::Batcher;
use burn::tensor::{Bool, Device, Int, Tensor, TensorData};

use chess_again_core::{Board, Color};
use chess_again_dataset::{N_MOVE_INDICES, RECORD_RSM_SIZE, RecordRSM, encode_move};
use chess_again_network::N_POLICY_CLASSES;

const N_SQUARES: usize = 64;
const N_INPUT_FEATURES: usize = 20;

// The policy vector is defined once by the model geometry; keep the dataset's
// index scheme in sync with it.
const _: () = assert!(N_MOVE_INDICES == N_POLICY_CLASSES);

#[derive(Debug, Clone)]
pub struct V1Batch {
    // input of size [B, 64, 20]
    pub input: Tensor<3>,
    // policy illegal mask of size [B, 4352]: true where the index is not a
    // legal move, so masked logits are excluded from the policy loss
    pub policy_illegal_mask: Tensor<2, Bool>,
    pub policy_targets: Tensor<1, Int>,
    pub wdl_targets: Tensor<1, Int>,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct V1Batcher;

impl Batcher<RecordRSM, V1Batch> for V1Batcher {
    fn batch(&self, records: Vec<RecordRSM>, device: &Device) -> V1Batch {
        let batch_size = records.len();
        let mut inputs = Vec::with_capacity(batch_size * N_SQUARES * N_INPUT_FEATURES);
        let mut policy_illegal_masks = vec![true; batch_size * N_POLICY_CLASSES];
        let mut policy_targets: Vec<i32> = Vec::with_capacity(batch_size);
        let mut wdl_targets: Vec<i32> = Vec::with_capacity(batch_size);

        for (batch_index, record) in records.into_iter().enumerate() {
            let (board, _, next_move, _) = record
                .to_board_force_white()
                .expect("training records must decode successfully");
            // The normalized board has white to move, which is how the model
            // and the policy indices are oriented.
            inputs.extend(encode_input(&board, record.bytes));
            mark_legal_moves(&board, batch_index, &mut policy_illegal_masks);
            policy_targets.push(encode_move(next_move) as i32);
            wdl_targets.push((record.bytes[33] & 0b11) as i32);
        }

        V1Batch {
            input: Tensor::from_data(
                TensorData::new(inputs, [batch_size, N_SQUARES, N_INPUT_FEATURES]),
                device,
            ),
            policy_illegal_mask: Tensor::from_data(
                TensorData::new(policy_illegal_masks, [batch_size, N_POLICY_CLASSES]),
                device,
            ),
            policy_targets: Tensor::from_data(
                TensorData::new(policy_targets, [batch_size]),
                device,
            ),
            wdl_targets: Tensor::from_data(TensorData::new(wdl_targets, [batch_size]), device),
        }
    }
}

fn encode_input(board: &Board, bytes: [u8; RECORD_RSM_SIZE]) -> Vec<f32> {
    let mut input = vec![0.0; N_SQUARES * N_INPUT_FEATURES];
    for (square, piece) in board.pieces() {
        let color_offset = match piece.color {
            Color::White => 0,
            Color::Black => 6,
        };
        input[square.index() * N_INPUT_FEATURES + color_offset + piece.piece_type as usize] = 1.0;
    }
    if let Some(square) = board.ep_square() {
        input[square.index() * N_INPUT_FEATURES + 12] = 1.0;
    }

    let castling = board.castling_rights();
    let repetition = ((bytes[33] >> 2) & 0b11) as f32 / 2.0;
    let rule50 = board.rule50() as f32 / 100.0;
    for square in 0..N_SQUARES {
        let offset = square * N_INPUT_FEATURES;
        input[offset + 13] = 1.0;
        input[offset + 14] = castling.kingside(Color::White) as u8 as f32;
        input[offset + 15] = castling.queenside(Color::White) as u8 as f32;
        input[offset + 16] = castling.kingside(Color::Black) as u8 as f32;
        input[offset + 17] = castling.queenside(Color::Black) as u8 as f32;
        input[offset + 18] = repetition;
        input[offset + 19] = rule50;
    }
    input
}

fn mark_legal_moves(board: &Board, batch_index: usize, policy_illegal_masks: &mut [bool]) {
    for mv in board.legal_moves() {
        let policy_index = encode_move(mv.into());
        policy_illegal_masks[batch_index * N_POLICY_CLASSES + policy_index] = false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chess_again_core::{GameResult, Move, Square};
    use chess_again_dataset::N_NON_PROMOTION_MOVE_INDICES;

    #[test]
    fn batches_normalized_record_targets_and_legal_masks() {
        let board = Board::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("e5", &board).expect("e5 should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::WhiteWin);

        let device = chess_again_network::default_device();
        let batch: V1Batch = V1Batcher.batch(vec![record], &device);

        assert_eq!(batch.input.dims(), [1, N_SQUARES, N_INPUT_FEATURES]);
        assert_eq!(batch.policy_illegal_mask.dims(), [1, N_POLICY_CLASSES]);
        // Black's e5 is stored mirrored: white's e2-e4 is the training target.
        assert_eq!(
            batch.policy_targets.to_data().try_to_vec::<i32>().unwrap(),
            [(Square::from_algebraic("e2").unwrap().index() * N_SQUARES
                + Square::from_algebraic("e4").unwrap().index()) as i32]
        );
        assert_eq!(
            batch.wdl_targets.to_data().try_to_vec::<i32>().unwrap(),
            [2]
        );

        let policy_mask = batch
            .policy_illegal_mask
            .to_data()
            .try_to_vec::<bool>()
            .unwrap();
        assert!(
            !policy_mask[Square::from_algebraic("e2").unwrap().index() * N_SQUARES
                + Square::from_algebraic("e4").unwrap().index()]
        );
        assert!(
            policy_mask[Square::from_algebraic("e2").unwrap().index() * N_SQUARES
                + Square::from_algebraic("e5").unwrap().index()]
        );
    }

    #[test]
    fn promotion_targets_use_the_flattened_promotion_indices() {
        let board = Board::from_fen("4k3/7P/8/8/8/8/8/4K3 w - - 0 1")
            .expect("test position should be valid");
        let mv = Move::from_san("h8=Q", &board).expect("h8=Q should be legal");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);

        let device = chess_again_network::default_device();
        let batch: V1Batch = V1Batcher.batch(vec![record], &device);

        // h7-h8 queen: index 4096 + (from_file * 8 + to_file) * 4 + queen.
        assert_eq!(
            batch.policy_targets.to_data().try_to_vec::<i32>().unwrap(),
            [(N_NON_PROMOTION_MOVE_INDICES + (7 * 8 + 7) * 4 + 3) as i32]
        );
        let policy_mask = batch
            .policy_illegal_mask
            .to_data()
            .try_to_vec::<bool>()
            .unwrap();
        assert!(!policy_mask[N_NON_PROMOTION_MOVE_INDICES + (7 * 8 + 7) * 4 + 3]);
        assert_eq!(
            policy_mask.iter().filter(|illegal| !**illegal).count(),
            board.legal_moves().len()
        );
    }
}
