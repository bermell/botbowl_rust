//! Minimal hand-rolled NumPy `.npy` v1.0 writer (+ reader for tests).
//!
//! Just enough of the [format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
//! to serialise the flat `f32` / `i64` arrays the prepare step emits —
//! avoids pulling the `ndarray` + `ndarray-npy` stack purely for I/O.
//! Always little-endian, C order.

use std::fs::File;
use std::io::{self, BufWriter, Read, Write};
use std::path::Path;

const MAGIC: &[u8] = b"\x93NUMPY";

/// Little-endian dtype descriptor strings.
pub const F4: &str = "<f4";
pub const I8: &str = "<i8";

fn header_bytes(descr: &str, shape: &[usize]) -> Vec<u8> {
    let shape_str = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => {
            let inner = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
            format!("({inner})")
        }
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Total = 6 (magic) + 2 (version) + 2 (header len) + dict + '\n',
    // padded with spaces so the whole thing is a multiple of 64.
    let prefix = MAGIC.len() + 2 + 2;
    let unpadded = prefix + dict.len() + 1;
    let pad = (64 - (unpadded % 64)) % 64;
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat(b' ').take(pad));
    header.push(b'\n');
    header
}

/// Elements converted per `write_all`. 8192 x 8 bytes = 64 KB of stack
/// scratch — big enough that syscall overhead is irrelevant, small enough
/// to be free.
const CHUNK_ELEMS: usize = 8192;

fn write_header(w: &mut impl Write, descr: &str, shape: &[usize]) -> io::Result<()> {
    let header = header_bytes(descr, shape);
    w.write_all(MAGIC)?;
    w.write_all(&[0x01, 0x00])?; // version 1.0
    let len = u16::try_from(header.len()).expect("npy header exceeds v1.0 u16 limit");
    w.write_all(&len.to_le_bytes())?;
    w.write_all(&header)
}

fn write_raw(path: impl AsRef<Path>, descr: &str, shape: &[usize], data: &[u8]) -> io::Result<()> {
    let mut w = BufWriter::new(File::create(path)?);
    write_header(&mut w, descr, shape)?;
    w.write_all(data)?;
    w.flush()
}

/// Write a flat `f32` buffer with the given shape (product must equal len).
///
/// Converts to little-endian bytes **in fixed-size chunks** rather than
/// materialising the whole array as a second `Vec<u8>`. That doubling is
/// not academic at this scale: a 3-generation training window is ~353k
/// samples x 37x9x16 f32 = 7.5 GB of `spatial`, so the old writer peaked
/// at ~14.4 GB and the kernel OOM-killed `prepare` on 2026-09-05
/// ("Killed process 958750 (prepare) anon-rss:14077600kB") on a 15.9 GB
/// box. Re-running it idle peaked at 14.37 GB against 14.36 GB available —
/// it had been fitting with no margin since the window landed.
///
/// Byte-for-byte identical output; only the peak allocation changes.
pub fn write_f32(path: impl AsRef<Path>, data: &[f32], shape: &[usize]) -> io::Result<()> {
    debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/len mismatch");
    let mut w = BufWriter::new(File::create(path)?);
    write_header(&mut w, F4, shape)?;
    let mut buf = [0u8; CHUNK_ELEMS * 4];
    for chunk in data.chunks(CHUNK_ELEMS) {
        for (i, v) in chunk.iter().enumerate() {
            buf[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf[..chunk.len() * 4])?;
    }
    w.flush()
}

/// Write a flat `i64` buffer with the given shape.
/// Write a flat `i64` buffer. Chunked for the same reason as
/// [`write_f32`] — `actions.npy` is `(M, 4)` with M ~5.6M rows.
pub fn write_i64(path: impl AsRef<Path>, data: &[i64], shape: &[usize]) -> io::Result<()> {
    debug_assert_eq!(shape.iter().product::<usize>(), data.len(), "shape/len mismatch");
    let mut w = BufWriter::new(File::create(path)?);
    write_header(&mut w, I8, shape)?;
    let mut buf = [0u8; CHUNK_ELEMS * 8];
    for chunk in data.chunks(CHUNK_ELEMS) {
        for (i, v) in chunk.iter().enumerate() {
            buf[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf[..chunk.len() * 8])?;
    }
    w.flush()
}

/// A parsed `.npy` file: dtype descriptor, shape, and raw payload bytes.
#[derive(Debug, Clone)]
pub struct Npy {
    pub descr: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl Npy {
    /// Decode the payload as `f32` (panics if the descriptor isn't `<f4`).
    pub fn as_f32(&self) -> Vec<f32> {
        assert_eq!(self.descr, F4, "expected <f4, got {}", self.descr);
        self.data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    /// Decode the payload as `i64` (panics if the descriptor isn't `<i8`).
    pub fn as_i64(&self) -> Vec<i64> {
        assert_eq!(self.descr, I8, "expected <i8, got {}", self.descr);
        self.data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
            .collect()
    }
}

/// Read and parse a `.npy` v1.0 file. Tolerant enough for our own output;
/// not a general-purpose parser.
pub fn read(path: impl AsRef<Path>) -> io::Result<Npy> {
    let mut buf = Vec::new();
    File::open(path)?.read_to_end(&mut buf)?;
    parse(&buf)
}

/// Parse `.npy` v1.0 bytes already in memory (used for the canary
/// fixture, which is `include_bytes!`-embedded in the remote client).
pub fn parse(buf: &[u8]) -> io::Result<Npy> {
    if buf.len() < 10 || &buf[..6] != MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "not a .npy file"));
    }
    let header_len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
    let header_end = 10 + header_len;
    let header =
        std::str::from_utf8(&buf[10..header_end]).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    let descr = extract(header, "'descr': '").unwrap_or_default();
    let shape = extract_shape(header);
    let data = buf[header_end..].to_vec();
    Ok(Npy { descr, shape, data })
}

fn extract(header: &str, key: &str) -> Option<String> {
    let start = header.find(key)? + key.len();
    let rest = &header[start..];
    let end = rest.find('\'')?;
    Some(rest[..end].to_string())
}

fn extract_shape(header: &str) -> Vec<usize> {
    let start = match header.find("'shape': (") {
        Some(s) => s + "'shape': (".len(),
        None => return Vec::new(),
    };
    let rest = &header[start..];
    let end = rest.find(')').unwrap_or(rest.len());
    rest[..end]
        .split(',')
        .filter_map(|t| t.trim().parse::<usize>().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("botbowl_nn_npy_{}_{}.npy", std::process::id(), name))
    }

    #[test]
    fn f32_round_trip_with_shape() {
        let path = tmp("f32");
        let data: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0, 100.0, -0.5];
        write_f32(&path, &data, &[2, 3]).unwrap();
        let back = read(&path).unwrap();
        assert_eq!(back.descr, F4);
        assert_eq!(back.shape, vec![2, 3]);
        assert_eq!(back.as_f32(), data);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn i64_round_trip_1d_and_2d() {
        let p1 = tmp("i64_1d");
        write_i64(&p1, &[0, 5, 10, 15], &[4]).unwrap();
        let b1 = read(&p1).unwrap();
        assert_eq!(b1.shape, vec![4]);
        assert_eq!(b1.as_i64(), vec![0, 5, 10, 15]);

        let p2 = tmp("i64_2d");
        write_i64(&p2, &[1, 2, 3, 4, 5, 6], &[3, 2]).unwrap();
        let b2 = read(&p2).unwrap();
        assert_eq!(b2.shape, vec![3, 2]);
        assert_eq!(b2.as_i64(), vec![1, 2, 3, 4, 5, 6]);
        std::fs::remove_file(&p1).ok();
        std::fs::remove_file(&p2).ok();
    }

    #[test]
    fn header_is_64_byte_aligned() {
        let h = header_bytes(F4, &[2, 3, 4]);
        let total = MAGIC.len() + 2 + 2 + h.len();
        assert_eq!(total % 64, 0);
        assert_eq!(*h.last().unwrap(), b'\n');
    }
}
