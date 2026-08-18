pub fn parse_algebraic(square: &str) -> Result<usize, String> {
    if square.len() != 2 {
        return Err(format!("Invalid square: '{}'", square));
    }
    let mut chars = square.chars();
    let file = chars.next().unwrap();
    let rank = chars.next().unwrap();
    if !('a'..='h').contains(&file) || !('1'..='8').contains(&rank) {
        return Err(format!("Invalid square: '{}'", square));
    }
    let file_index = (file as u8 - b'a') as usize;
    let rank_index = (rank as u8 - b'1') as usize;
    Ok(rank_index * 8 + file_index)
}