//! Minimal hand-rolled NumPy `.npy` v1.0 writer (+ reader for tests).
//!
//! Just enough of the [format](https://numpy.org/doc/stable/reference/generated/numpy.lib.format.html)
//! to serialise the flat `f32` / `i64` arrays the prepare step emits —
//! avoids pulling the `ndarray` + `ndarray-npy` stack purely for I/O.
//! Always little-endian, C order.

use std::fs::File;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::marker::PhantomData;
use std::path::Path;

const MAGIC: &[u8] = b"\x93NUMPY";

/// Bytes before the dict: 6 (magic) + 2 (version) + 2 (header len).
const PREFIX: usize = MAGIC.len() + 2 + 2;

/// Little-endian dtype descriptor strings.
pub const F4: &str = "<f4";
pub const I8: &str = "<i8";

fn header_bytes(descr: &str, shape: &[usize]) -> Vec<u8> {
    header_bytes_min(descr, shape, 0)
}

/// The v1.0 header dict, space-padded so that `PREFIX + result.len()` is a
/// multiple of 64 **and** at least `min_total`. `min_total = 0` gives the
/// plain format rule; [`StreamWriter`] passes [`STREAM_HEADER_RESERVE`] so the
/// header it back-patches exactly fills the region it reserved up front.
fn header_bytes_min(descr: &str, shape: &[usize], min_total: usize) -> Vec<u8> {
    let shape_str = match shape {
        [] => "()".to_string(),
        [n] => format!("({n},)"),
        _ => {
            let inner = shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", ");
            format!("({inner})")
        }
    };
    let dict = format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_str}, }}");
    // Total = PREFIX + dict + '\n', padded with spaces to a multiple of 64.
    let unpadded = PREFIX + dict.len() + 1;
    let target = std::cmp::max(unpadded.div_ceil(64) * 64, min_total);
    let mut header = dict.into_bytes();
    header.extend(std::iter::repeat(b' ').take(target - unpadded));
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

/// A scalar type this module can serialise. Lets [`StreamWriter`] be generic
/// over `f32`/`i64` without a runtime dtype tag that `push` would have to check.
pub trait NpyScalar: Copy {
    const DESCR: &'static str;
    const SIZE: usize;
    fn write_le(self, out: &mut [u8]);
}

impl NpyScalar for f32 {
    const DESCR: &'static str = F4;
    const SIZE: usize = 4;
    fn write_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }
}

impl NpyScalar for i64 {
    const DESCR: &'static str = I8;
    const SIZE: usize = 8;
    fn write_le(self, out: &mut [u8]) {
        out.copy_from_slice(&self.to_le_bytes());
    }
}

/// Bytes reserved at the head of a streamed file for its header.
///
/// The v1.0 header is padded to a multiple of 64 and its length depends only
/// on dtype + shape — never on the data — so reserving a fixed region and
/// back-patching it once `N` is known is exact, not a guess. 128 covers every
/// shape `prepare` emits (the widest, `(N, 37, 9, 16)`, needs 74 bytes before
/// padding), and [`StreamWriter::finish`] asserts rather than silently
/// corrupting a file if that ever stops being true.
pub const STREAM_HEADER_RESERVE: usize = 128;

/// Append-only `.npy` writer whose **leading axis length is discovered as it
/// goes**: reserve [`STREAM_HEADER_RESERVE`] bytes, stream rows straight to
/// the file, then seek back and write the real header.
///
/// This exists so `prepare` does not have to hold the whole prepared corpus in
/// RAM before it knows `N`. `spatial` alone is 21,312 bytes/sample, so a
/// 7-generation window (~825k samples) would be ~17.6 GB of `Vec<f32>` on a
/// 14.4 GB box; the kernel already OOM-killed `prepare` once at a 3-generation
/// window. Streaming makes peak RSS O(1) in corpus size for a single parse
/// (a two-pass "count then write" would double the ~27 s parse instead).
///
/// Output is byte-identical to [`write_f32`] / [`write_i64`] for any shape
/// whose natural header is already `STREAM_HEADER_RESERVE` bytes — which is
/// every array `prepare` writes. For a shape small enough to fit a 64-byte
/// header the streamed file carries 64 extra padding spaces; still a valid
/// `.npy`, just not byte-equal to the buffered writer's.
pub struct StreamWriter<T: NpyScalar> {
    file: BufWriter<File>,
    /// Shape dims *after* the leading axis (empty for a 1-D array).
    trailing: Vec<usize>,
    /// Product of `trailing`; 1 for a 1-D array.
    row_elems: usize,
    elems: usize,
    /// Reused LE-conversion scratch, so `push` doesn't re-zero 64 KB per call.
    scratch: Vec<u8>,
    _t: PhantomData<T>,
}

impl<T: NpyScalar> StreamWriter<T> {
    /// Create `path` and reserve its header region. `trailing` is the shape
    /// minus the leading axis, e.g. `&[37, 9, 16]` for `(N, 37, 9, 16)` or
    /// `&[]` for `(N,)`.
    pub fn create(path: impl AsRef<Path>, trailing: &[usize]) -> io::Result<Self> {
        let mut file = BufWriter::new(File::create(path)?);
        file.write_all(&[0u8; STREAM_HEADER_RESERVE])?;
        Ok(Self {
            file,
            trailing: trailing.to_vec(),
            row_elems: trailing.iter().product::<usize>().max(1),
            elems: 0,
            scratch: vec![0u8; CHUNK_ELEMS * T::SIZE],
            _t: PhantomData,
        })
    }

    /// Append elements in C order.
    pub fn push(&mut self, data: &[T]) -> io::Result<()> {
        for chunk in data.chunks(CHUNK_ELEMS) {
            for (i, v) in chunk.iter().enumerate() {
                v.write_le(&mut self.scratch[i * T::SIZE..(i + 1) * T::SIZE]);
            }
            self.file.write_all(&self.scratch[..chunk.len() * T::SIZE])?;
        }
        self.elems += data.len();
        Ok(())
    }

    /// Number of leading-axis rows written so far.
    pub fn rows(&self) -> usize {
        self.elems / self.row_elems
    }

    /// Back-patch the header and close. Returns the leading-axis length.
    pub fn finish(mut self) -> io::Result<usize> {
        assert_eq!(
            self.elems % self.row_elems,
            0,
            "npy stream: {} elements is not a whole number of {}-element rows",
            self.elems,
            self.row_elems
        );
        let n = self.elems / self.row_elems;
        let mut shape = Vec::with_capacity(self.trailing.len() + 1);
        shape.push(n);
        shape.extend_from_slice(&self.trailing);

        self.file.flush()?;
        let mut file = self.file.into_inner().map_err(|e| e.into_error())?;
        let header = header_bytes_min(T::DESCR, &shape, STREAM_HEADER_RESERVE);
        assert_eq!(
            PREFIX + header.len(),
            STREAM_HEADER_RESERVE,
            "npy stream: header for shape {shape:?} does not fit the {STREAM_HEADER_RESERVE}-byte reservation"
        );
        file.seek(SeekFrom::Start(0))?;
        file.write_all(MAGIC)?;
        file.write_all(&[0x01, 0x00])?;
        let len = u16::try_from(header.len()).expect("npy header exceeds v1.0 u16 limit");
        file.write_all(&len.to_le_bytes())?;
        file.write_all(&header)?;
        file.flush()?;
        Ok(n)
    }
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

    #[test]
    fn stream_writer_round_trips_and_back_patches_n() {
        let path = tmp("stream_f32");
        let mut w = StreamWriter::<f32>::create(&path, &[2, 3]).unwrap();
        w.push(&[1.0, -2.5, 3.25, 0.0, 100.0, -0.5]).unwrap();
        assert_eq!(w.rows(), 1);
        w.push(&[7.0; 6]).unwrap();
        assert_eq!(w.finish().unwrap(), 2);

        let back = read(&path).unwrap();
        assert_eq!(back.descr, F4);
        assert_eq!(back.shape, vec![2, 2, 3]);
        assert_eq!(back.as_f32(), vec![1.0, -2.5, 3.25, 0.0, 100.0, -0.5, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stream_writer_handles_1d_and_i64() {
        let path = tmp("stream_i64");
        let mut w = StreamWriter::<i64>::create(&path, &[]).unwrap();
        for v in 0..5i64 {
            w.push(&[v * 10]).unwrap();
        }
        assert_eq!(w.finish().unwrap(), 5);
        let back = read(&path).unwrap();
        assert_eq!(back.shape, vec![5]);
        assert_eq!(back.as_i64(), vec![0, 10, 20, 30, 40]);
        std::fs::remove_file(&path).ok();
    }

    /// The whole point of the reservation: for the shapes `prepare` actually
    /// emits, the streamed file must be byte-for-byte what the buffered writer
    /// produced, so a regenerated corpus md5-matches an existing one.
    #[test]
    fn stream_writer_is_byte_identical_to_the_buffered_writer() {
        // (N, 37, 9, 16) is the real `spatial` shape; its header pads to 128.
        let rows = 3usize;
        let row = 37 * 9 * 16;
        let data: Vec<f32> = (0..rows * row).map(|i| i as f32 * 0.5 - 1.0).collect();

        let buffered = tmp("buffered_spatial");
        write_f32(&buffered, &data, &[rows, 37, 9, 16]).unwrap();

        let streamed = tmp("streamed_spatial");
        let mut w = StreamWriter::<f32>::create(&streamed, &[37, 9, 16]).unwrap();
        for r in data.chunks(row) {
            w.push(r).unwrap();
        }
        w.finish().unwrap();

        assert_eq!(std::fs::read(&buffered).unwrap(), std::fs::read(&streamed).unwrap());
        std::fs::remove_file(&buffered).ok();
        std::fs::remove_file(&streamed).ok();
    }
}
