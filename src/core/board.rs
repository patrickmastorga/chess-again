#[derive(Clone, Debug)]
pub struct Board {
    pub eval: i32,
}

impl Board {
    pub fn new() -> Self {
        Self {
            eval: 42,
        }
    }

    pub fn evaluate(&self) -> i32 {
        self.eval
    }
}