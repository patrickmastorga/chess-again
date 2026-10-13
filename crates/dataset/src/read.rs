use std::io::{ErrorKind, Read};

use anyhow::{Context, Result};

/// Fills `buf` with as many bytes as the decoder will give, stopping early
/// only at end of stream.
///
/// Returns the number of bytes written, which is always `buf.len()` unless the
/// stream ended first
pub fn read_chunk<R: Read>(decoder: &mut R, buf: &mut [u8]) -> Result<usize> {
    let mut filled = 0;
    while filled < buf.len() {
        match decoder.read(&mut buf[filled..]) {
            Ok(0) => break,
            Ok(n) => filled += n,
            Err(error) if error.kind() == ErrorKind::Interrupted => continue,
            Err(error) => {
                return Err(error).context("failed to read from zstd decoder");
            }
        }
    }
    Ok(filled)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A reader that hands out one byte at a time and raises `Interrupted`
    /// before every other byte, to prove both are handled.
    struct Dribble {
        data: Vec<u8>,
        position: usize,
        interrupt_next: bool,
    }

    impl Dribble {
        fn new(data: &[u8]) -> Self {
            Self {
                data: data.to_vec(),
                position: 0,
                interrupt_next: true,
            }
        }
    }

    impl Read for Dribble {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            if buf.is_empty() {
                return Ok(0);
            }
            if self.interrupt_next {
                self.interrupt_next = false;
                return Err(std::io::Error::from(ErrorKind::Interrupted));
            }
            self.interrupt_next = true;
            if self.position == self.data.len() {
                return Ok(0);
            }
            buf[0] = self.data[self.position];
            self.position += 1;
            Ok(1)
        }
    }

    #[test]
    fn fills_the_whole_buffer() {
        let mut source = Cursor::new(b"abcdefgh".to_vec());
        let mut buf = [0u8; 8];
        assert_eq!(read_chunk(&mut source, &mut buf).unwrap(), 8);
        assert_eq!(&buf, b"abcdefgh");
    }

    #[test]
    fn reports_a_short_fill_at_end_of_stream() {
        let mut source = Cursor::new(b"abc".to_vec());
        let mut buf = [0u8; 8];
        assert_eq!(read_chunk(&mut source, &mut buf).unwrap(), 3);
        assert_eq!(&buf[..3], b"abc");

        // A stream that is already exhausted reports a zero-length fill.
        assert_eq!(read_chunk(&mut source, &mut buf).unwrap(), 0);
    }

    #[test]
    fn retries_interrupted_reads() {
        let mut source = Dribble::new(b"abcd");
        let mut buf = [0u8; 4];
        assert_eq!(read_chunk(&mut source, &mut buf).unwrap(), 4);
        assert_eq!(&buf, b"abcd");
    }

    #[test]
    fn propagates_other_errors() {
        struct Failing;
        impl Read for Failing {
            fn read(&mut self, _buf: &mut [u8]) -> std::io::Result<usize> {
                Err(std::io::Error::from(ErrorKind::PermissionDenied))
            }
        }
        let mut buf = [0u8; 4];
        let error = read_chunk(&mut Failing, &mut buf).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("failed to read from zstd decoder")
        );
    }
}
