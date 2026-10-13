//! End-to-end check that a freshly written RecordRSM shard validates and
//! streams back through the real loader.

use chess_again_core::{Board, GameResult, Move};
use chess_again_dataset::validate_dataset;
use chess_again_dataset::{RECORD_RSM_SIZE, RecordRSM};

#[test]
fn freshly_written_rsm_shard_validates() {
    let dir = std::env::temp_dir().join(format!(
        "chess-again-rsm-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // Real play, including castling, with both colours to move.
    let mut board = Board::starting();
    let mut raw = Vec::new();
    let mut expected = 0usize;
    for san in [
        "e4", "e5", "Nf3", "Nc6", "Bb5", "a6", "Ba4", "Nf6", "O-O", "Be7",
    ] {
        let mv = Move::from_san(san, &board).expect("legal san");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::Draw);
        assert_eq!(record.bytes.len(), RECORD_RSM_SIZE);

        // The record decodes back to this position and move. Only the first
        // four FEN fields are compared: the halfmove clock is reconstructed
        // from rule50, which is all the record stores.
        let (decoded, decoded_move, _) = record.to_board().expect("record decodes");
        let fen_prefix = |fen: &str| fen.split(' ').take(4).collect::<Vec<_>>().join(" ");
        assert_eq!(fen_prefix(&decoded.to_fen()), fen_prefix(&board.to_fen()));
        assert_eq!(decoded_move, mv.into());

        raw.extend_from_slice(&record.bytes);
        expected += 1;
        board.make_move_unchecked(mv);
    }

    let compressed = zstd::stream::encode_all(raw.as_slice(), 1).expect("compress");
    std::fs::write(dir.join("shard-000000.zst"), compressed).expect("write shard");
    std::fs::write(
        dir.join("manifest.csv"),
        format!("filename,records\nshard-000000.zst,{expected}\n"),
    )
    .expect("write manifest");

    let report = validate_dataset::<RecordRSM>(&dir, 2).expect("validation runs");
    assert!(
        report.is_valid(),
        "fresh RSM shard should validate: {:?}",
        report.errors
    );
    assert_eq!(report.records_checked, expected);

    std::fs::remove_dir_all(dir).expect("cleanup");
}

#[test]
fn freshly_written_promotion_records_validate() {
    let dir = std::env::temp_dir().join(format!(
        "chess-again-rsm-promo-{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock after epoch")
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).expect("temp dir");

    // A position where both sides can promote, exercised for each colour.
    let mut raw = Vec::new();
    let mut expected = 0usize;
    for (fen, san) in [
        ("4k3/7P/8/8/8/8/8/4K3 w - - 0 1", "h8=Q"),
        ("4k3/7P/8/8/8/8/8/4K3 w - - 0 1", "h8=N"),
        ("4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h1=Q"),
        ("4k3/8/8/8/8/8/7p/4K3 b - - 0 1", "h1=R"),
    ] {
        let board = Board::from_fen(fen).expect("valid fen");
        let mv = Move::from_san(san, &board).expect("legal san");
        let record = RecordRSM::from_board(&board, mv.into(), GameResult::WhiteWin);
        raw.extend_from_slice(&record.bytes);
        expected += 1;
    }

    eprintln!("raw len = {}, expected = {}", raw.len(), expected);
    assert_eq!(raw.len(), expected * RECORD_RSM_SIZE);
    let compressed = zstd::stream::encode_all(raw.as_slice(), 1).expect("compress");
    std::fs::write(dir.join("shard-000000.zst"), compressed).expect("write shard");
    std::fs::write(
        dir.join("manifest.csv"),
        format!("filename,records\nshard-000000.zst,{expected}\n"),
    )
    .expect("write manifest");

    let report = validate_dataset::<RecordRSM>(&dir, 2).expect("validation runs");
    assert!(
        report.is_valid(),
        "promotion records should validate: {:?}",
        report.errors
    );
    assert_eq!(report.records_checked, expected);

    std::fs::remove_dir_all(dir).expect("cleanup");
}
