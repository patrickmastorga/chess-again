use std::ffi::OsStr;
use std::io::{self, BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, channel};
use std::thread;
use std::time::{Duration, Instant};

use crate::engine::{Engine, Score, SearchResult, UciMove};

const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, Clone)]
pub enum UciEvent {
    BestMove {
        best: UciMove,
        ponder: Option<UciMove>,
    },
    Info(Info),
    Other(String),
}

#[derive(Debug, Clone, Default)]
pub struct Info {
    pub multipv: Option<u32>,
    pub depth: Option<u32>,
    pub score: Option<Score>,
    pub nodes: Option<u64>,
    pub pv: Vec<UciMove>,
}

pub enum Go {
    Depth(u32),
    Movetime(u64), // ms
    Infinite,      // keep going until stop(); bestmove arrives then
}

pub struct UciEngine {
    pub name: String,
    pub author: String,
    child: Child,
    stdin: ChildStdin,
    stdout_rx: Receiver<String>,
    /// Events produced by the current search, cleared when one starts.
    events: Vec<UciEvent>,
    /// Whether a `go` is outstanding.
    searching: bool,
}

impl UciEngine {
    pub fn new(program: impl AsRef<OsStr>) -> io::Result<Self> {
        let mut child = Command::new(program)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;

        let stdin = child.stdin.take().expect("stdin was piped");
        let stdout = child.stdout.take().expect("stdout was piped");

        // Dedicated reader thread: stdout MUST be drained continuously or the
        // engine deadlocks once its pipe buffer (~64 KiB) fills up.
        let (tx, rx) = channel();
        thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                match line {
                    Ok(line) => {
                        if tx.send(line).is_err() {
                            break;
                        }
                    }
                    Err(_) => break, // engine exited
                }
            }
        });

        let mut engine = UciEngine {
            name: "unknown".to_owned(),
            author: "unknown".to_owned(),
            child,
            stdin,
            stdout_rx: rx,
            events: Vec::new(),
            searching: false,
        };
        engine.send_line("uci")?;
        engine.wait_for_uciok(HANDSHAKE_TIMEOUT)?;
        Ok(engine)
    }

    pub fn is_alive(&mut self) -> bool {
        matches!(self.child.try_wait(), Ok(None))
    }

    /// Block until the engine is idle. This is your synchronization primitive:
    /// call it after ucinewgame/setoption/position and before reading results.
    pub fn sync(&mut self) -> io::Result<()> {
        self.send_line("isready")?;
        self.wait_for_line("readyok", HANDSHAKE_TIMEOUT)
    }

    pub fn send_new_game(&mut self) -> io::Result<()> {
        self.send_line("ucinewgame")?;
        self.sync()
    }

    pub fn set_option(&mut self, name: &str, value: &str) -> io::Result<()> {
        self.send_line(format!("setoption name {name} value {value}"))?;
        self.sync()
    }

    /// `moves` are UCI strings like "e2e4", applied after the FEN.
    pub fn send_position(&mut self, fen: &str, moves: &[UciMove]) -> io::Result<()> {
        let mut cmd = format!("position fen {fen}");
        if !moves.is_empty() {
            cmd.push_str(" moves");
            for m in moves {
                cmd.push(' ');
                cmd.push_str(m.to_string().as_str());
            }
        }
        self.send_line(cmd)?;
        self.sync()
    }

    /// Starts a search.
    pub fn send_go(&mut self, limit: Go) -> io::Result<()> {
        self.send_line(match limit {
            Go::Depth(d) => format!("go depth {d}"),
            Go::Movetime(ms) => format!("go movetime {ms}"),
            Go::Infinite => "go infinite".to_owned(),
        })
    }

    pub fn send_stop(&mut self) -> io::Result<()> {
        self.send_line("stop")
    }

    /// Non-blocking drain of everything the engine produced since the last call.
    /// Call this every frame from your egui update().
    pub fn drain_events(&mut self) -> Vec<UciEvent> {
        let before = self.events.len();
        self.poll_events();
        self.events[before..].to_vec()
    }

    /// Drains stdout into the event buffer, settling `searching` on `bestmove`.
    fn poll_events(&mut self) {
        while let Ok(line) = self.stdout_rx.try_recv() {
            let event = parse_line(&line);
            if let UciEvent::BestMove { .. } = &event {
                self.searching = false;
            }
            self.events.push(event);
        }
    }

    /// Consumes the finished search's `bestmove` and the eval that led to it.
    fn take_result(&mut self) -> Option<SearchResult> {
        let index = self
            .events
            .iter()
            .position(|event| matches!(event, UciEvent::BestMove { .. }))?;
        let UciEvent::BestMove { best, .. } = self.events.remove(index) else {
            unreachable!("index selected a bestmove")
        };
        let eval = self.events.iter().rev().find_map(|event| match event {
            UciEvent::Info(info) => info.score,
            _ => None,
        });
        self.events.clear();
        Some(SearchResult {
            best_move: best,
            eval,
        })
    }

    fn send_line(&mut self, cmd: impl AsRef<str>) -> io::Result<()> {
        writeln!(self.stdin, "{}", cmd.as_ref())?; // UCI is LF-terminated lines
        self.stdin.flush()
    }

    fn wait_for_line(&mut self, prefix: &str, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            match self.stdout_rx.recv_timeout(remaining) {
                Ok(line) if line.starts_with(prefix) => return Ok(()),
                Ok(_) => {} // skip banners/other lines during handshake
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "engine terminated",
                    ));
                }
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            format!("timed out waiting for `{prefix}`"),
        ))
    }

    fn wait_for_uciok(&mut self, timeout: Duration) -> io::Result<()> {
        let deadline = Instant::now() + timeout;
        while let Some(remaining) = deadline.checked_duration_since(Instant::now()) {
            match self.stdout_rx.recv_timeout(remaining) {
                Ok(line) if line == "uciok" => return Ok(()),
                Ok(line) => {
                    if let Some(name) = line.strip_prefix("id name ") {
                        self.name = name.to_owned();
                    } else if let Some(author) = line.strip_prefix("id author ") {
                        self.author = author.to_owned();
                    }
                }
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "engine terminated",
                    ));
                }
                Err(RecvTimeoutError::Timeout) => break,
            }
        }
        Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "timed out waiting for `uciok`",
        ))
    }
}

impl Engine for UciEngine {
    fn start_search_async(&mut self, fen: &str, time: Option<u64>) -> Result<(), String> {
        self.events.clear();
        self.send_new_game().map_err(|error| error.to_string())?;
        self.send_position(fen, &[])
            .map_err(|error| error.to_string())?;
        let limit = match time {
            Some(ms) => Go::Movetime(ms),
            None => Go::Infinite,
        };
        self.send_go(limit).map_err(|error| error.to_string())?;
        self.searching = true;
        Ok(())
    }

    fn poll(&mut self) -> Result<Option<SearchResult>, String> {
        self.poll_events();
        Ok(self.take_result())
    }

    fn stop(&mut self) -> Result<Option<SearchResult>, String> {
        if !self.searching {
            return Ok(None);
        }
        self.send_stop().map_err(|error| error.to_string())?;
        // Block until the search is fully over so no reply can outlive it.
        while self.searching {
            self.poll_events();
        }
        Ok(self.take_result())
    }
}

impl Drop for UciEngine {
    fn drop(&mut self) {
        let _ = self.send_line("quit");
        let _ = self.child.kill();
        let _ = self.child.wait(); // reap the zombie
    }
}

fn parse_line(line: &str) -> UciEvent {
    if let Some(rest) = line.strip_prefix("bestmove ") {
        let mut words = rest.split_whitespace();
        let best =
            UciMove::from_string(words.next().expect("bestmove response must include a move"))
                .expect("bestmove response must contain a valid UCI move");
        let ponder = match (words.next(), words.next()) {
            (Some("ponder"), Some(move_text)) => UciMove::from_string(move_text).ok(),
            _ => None,
        };
        return UciEvent::BestMove { best, ponder };
    }
    if line.starts_with("info ") && line.contains(" pv ") {
        return UciEvent::Info(parse_info(line));
    }
    UciEvent::Other(line.to_owned())
}

fn parse_info(line: &str) -> Info {
    let mut info = Info::default();
    let w: Vec<&str> = line.split_whitespace().collect();
    let mut i = 0;
    while i < w.len() {
        match w[i] {
            "depth" => info.depth = w.get(i + 1).and_then(|t| t.parse().ok()),
            "multipv" => info.multipv = w.get(i + 1).and_then(|t| t.parse().ok()),
            "nodes" => info.nodes = w.get(i + 1).and_then(|t| t.parse().ok()),
            "score" => match w.get(i + 1).copied() {
                Some("cp") => info.score = w.get(i + 2).and_then(|t| t.parse().ok()).map(Score::Cp),
                Some("mate") => {
                    info.score = w.get(i + 2).and_then(|t| t.parse().ok()).map(Score::Mate)
                }
                _ => {}
            },
            "pv" => {
                // pv is the last field
                info.pv = w[i + 1..]
                    .iter()
                    .map_while(|move_text| UciMove::from_string(move_text).ok())
                    .collect();
                break;
            }
            _ => {}
        }
        i += 1;
    }
    info
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bestmove_and_ponder_as_uci_moves() {
        let UciEvent::BestMove { best, ponder } = parse_line("bestmove e2e4 ponder e7e5") else {
            panic!("expected bestmove event");
        };

        assert_eq!(best.to_string(), "e2e4");
        assert_eq!(ponder.map(|mv| mv.to_string()), Some("e7e5".to_string()));
    }

    #[test]
    fn retains_valid_pv_prefix_until_parse_error() {
        let info = parse_info("info depth 12 pv e2e4 e7e5 invalid g1f3");

        assert_eq!(info.depth, Some(12));
        assert_eq!(
            info.pv.iter().map(UciMove::to_string).collect::<Vec<_>>(),
            ["e2e4", "e7e5"]
        );
    }

    #[test]
    fn parses_the_null_move() {
        let UciEvent::BestMove { best, ponder } = parse_line("bestmove 0000") else {
            panic!("expected bestmove event");
        };

        assert_eq!(best, UciMove::NULL);
        assert_eq!(best.to_string(), "0000");
        assert_eq!(ponder, None);
    }

    #[test]
    #[should_panic(expected = "bestmove response must contain a valid UCI move")]
    fn panics_for_an_invalid_bestmove() {
        parse_line("bestmove invalid");
    }
}
