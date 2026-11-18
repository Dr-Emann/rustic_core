use std::io::Read;

use rand::{RngExt, rng};
use rustic_cdc::{Polynom, Polynom64};

use crate::error::{ErrorKind, RusticError, RusticResult};

mod scan;

use scan::CutFinder;

pub(super) mod constants {
    /// The size of a kilobyte.
    pub(super) const KB: usize = 1024;
    /// Lower bound for how many bytes are read and scanned for a chunk boundary at once.
    pub(super) const MIN_SCAN_SIZE: usize = 4 * KB;
    /// Upper bound for how many bytes are read and scanned for a chunk boundary at once.
    ///
    /// Data read past a boundary is copied into the next chunk, so this shouldn't be too large.
    pub(super) const MAX_SCAN_SIZE: usize = 64 * KB;
    /// Size of the rolling hash window, which the data of a chunk has to fill.
    pub(super) const WINDOW_SIZE: usize = 64;
    /// Random polynomial maximum tries.
    pub(super) const RAND_POLY_MAX_TRIES: i32 = 1_000_000;
}

pub(crate) fn check_rabin_params(
    chunk_size: usize,
    chunk_min_size: usize,
    chunk_max_size: usize,
) -> RusticResult<()> {
    if (chunk_size & (chunk_size - 1)) != 0 {
        return Err(RusticError::new(
            ErrorKind::Unsupported,
            "Chunk size must be a power of 2 for the rabin chunker. chunk size = {chunk_size}.",
        )
        .attach_context("chunk_size", chunk_size.to_string()));
    }
    if chunk_min_size < constants::WINDOW_SIZE {
        return Err(RusticError::new(
            ErrorKind::Unsupported,
            "Chunk min size must be at least the rolling hash window. window size = {window_size}.",
        )
        .attach_context("window_size", constants::WINDOW_SIZE.to_string()));
    }
    if chunk_min_size > chunk_size {
        return Err(RusticError::new(
            ErrorKind::Unsupported,
            "Chunk min size must be smaller or equal than the chunk size.",
        ));
    }
    if chunk_max_size < chunk_size {
        return Err(RusticError::new(
            ErrorKind::Unsupported,
            "Chunk max size must be larger or equal than the chunk size.",
        ));
    }
    Ok(())
}

/// `ChunkIter` is an iterator that chunks data.
pub(crate) struct ChunkIter<R: Read + Send> {
    /// Data read past the end of the previous chunk; the start of the next chunk.
    carry: Vec<u8>,

    /// The reader.
    reader: R,

    /// Finds the chunk boundaries.
    cut_finder: CutFinder,

    /// The size hint is used to optimize memory allocation; this should be an upper bound on the size.
    size_hint: usize,

    /// The minimum size of a chunk.
    min_size: usize,

    /// The maximum size of a chunk.
    max_size: usize,

    /// How many bytes to read and scan for a boundary at once.
    scan_size: usize,

    /// If the reader is exhausted.
    finished: bool,
}

impl<R: Read + Send> ChunkIter<R> {
    /// Creates a new `ChunkIter`.
    ///
    /// # Arguments
    ///
    /// * `poly` - The modulo polynomial of the rolling hash.
    /// * `reader` - The reader to read from.
    /// * `size_hint` - The size hint is used to optimize memory allocation; this should be an upper bound on the size.
    pub(crate) fn new(
        poly: Polynom64,
        chunk_size: usize,
        chunk_min_size: usize,
        chunk_max_size: usize,
        reader: R,
        size_hint: usize,
    ) -> RusticResult<Self> {
        check_rabin_params(chunk_size, chunk_min_size, chunk_max_size)?;
        let split_mask: u64 = (chunk_size - 1).try_into().unwrap();
        let Some(cut_finder) = CutFinder::new(poly, split_mask) else {
            return Err(RusticError::new(
                ErrorKind::Unsupported,
                "The rabin chunker polynomial must have a degree of at least 8. polynomial = {poly}.",
            )
            .attach_context("poly", format!("{poly:x}")));
        };
        Ok(Self {
            carry: Vec::new(),
            reader,
            cut_finder,
            size_hint,
            min_size: chunk_min_size,
            max_size: chunk_max_size,
            scan_size: (chunk_size / 4).clamp(constants::MIN_SCAN_SIZE, constants::MAX_SCAN_SIZE),
            finished: false,
        })
    }

    /// Reads from the reader until `chunk` holds `target_len` bytes or the reader is exhausted.
    fn fill(&mut self, chunk: &mut Vec<u8>, target_len: usize) -> RusticResult<()> {
        if self.finished || chunk.len() >= target_len {
            return Ok(());
        }
        let missing = target_len - chunk.len();
        let read = (&mut self.reader)
            .take(missing as u64)
            .read_to_end(chunk)
            .map_err(|err| {
                RusticError::with_source(
                    ErrorKind::InputOutput,
                    "Failed to read from reader in iterator",
                    err,
                )
            })?;
        if read < missing {
            self.finished = true;
        }
        Ok(())
    }

    /// Returns the length of the chunk at the start of `chunk`, reading more data into it as needed.
    ///
    /// On error, the data read so far is left in `chunk`.
    fn chunk_len(&mut self, chunk: &mut Vec<u8>) -> RusticResult<usize> {
        self.fill(chunk, self.min_size)?;
        if chunk.len() < self.min_size {
            return Ok(chunk.len());
        }

        let mut scan_from = self.min_size - 1;
        loop {
            let scan_end = chunk
                .len()
                .min(self.max_size)
                .min(scan_from + self.scan_size);
            if let Some(pos) = self.cut_finder.find_cut(chunk, scan_from..scan_end) {
                return Ok(pos + 1);
            }
            scan_from = scan_end;
            if scan_from == self.max_size {
                return Ok(self.max_size);
            }
            if scan_from == chunk.len() {
                self.fill(chunk, scan_from + self.scan_size)?;
                if chunk.len() == scan_from {
                    return Ok(scan_from);
                }
            }
        }
    }
}

impl<R: Read + Send> Iterator for ChunkIter<R> {
    type Item = RusticResult<Vec<u8>>;

    fn next(&mut self) -> Option<Self::Item> {
        let mut chunk = std::mem::take(&mut self.carry);
        chunk.reserve(self.size_hint.min(self.min_size).saturating_sub(chunk.len()));
        let len = match self.chunk_len(&mut chunk) {
            Ok(len) => len,
            Err(err) => {
                self.carry = chunk;
                return Some(Err(err));
            }
        };
        if len == 0 {
            return None;
        }
        self.carry = chunk.split_off(len);
        self.size_hint = self.size_hint.saturating_sub(len); // size_hint can be too small!
        Some(Ok(chunk))
    }
}

/// [`random_poly`] returns an random irreducible polynomial of degree 53
/// (largest prime number below 64-8)
/// There are (2^53-2/53) irreducible polynomials of degree 53 in
/// `F_2[X]`, c.f. Michael O. Rabin (1981): "Fingerprinting by Random
/// Polynomials", page 4.
///
/// # Errors
///
/// * If no polynomial could be found in one million tries.
pub fn random_poly() -> RusticResult<u64> {
    for _ in 0..constants::RAND_POLY_MAX_TRIES {
        let mut poly: u64 = rng().random();

        // mask away bits above bit 53
        poly &= (1 << 54) - 1;

        // set highest and lowest bit so that the degree is 53 and the
        // polynomial is not trivially reducible
        poly |= (1 << 53) | 1;

        if poly.irreducible() {
            return Ok(poly);
        }
    }

    Err(RusticError::new(
        ErrorKind::Internal,
        "No suitable polynomial found, this should essentially never happen. Please try again.",
    )
    .ask_report())
}

/// A trait for extending polynomials.
pub(crate) trait PolynomExtend {
    /// Returns true IFF x is irreducible over `F_2`.
    fn irreducible(&self) -> bool;

    /// Returns the degree of the polynomial.
    fn gcd(self, other: Self) -> Self;

    /// Adds two polynomials.
    fn add(self, other: Self) -> Self;

    /// Multiplies two polynomials modulo another polynomial.
    fn mulmod(self, other: Self, modulo: Self) -> Self;
}

// implementation goes along the lines of
// https://github.com/restic/chunker/blob/master/polynomials.go
impl PolynomExtend for Polynom64 {
    // Irreducible returns true IFF x is irreducible over F_2. This function
    // uses Ben Or's reducibility test.
    //
    // For details see "Tests and Constructions of Irreducible Polynomials over
    // Finite Fields".
    fn irreducible(&self) -> bool {
        for i in 1..=self.degree() / 2 {
            if self.gcd(qp(i, *self)) != 1 {
                return false;
            }
        }
        true
    }

    fn gcd(self, other: Self) -> Self {
        if other == 0 {
            return self;
        }

        if self == 0 {
            return other;
        }

        if self.degree() < other.degree() {
            self.gcd(other.modulo(&self))
        } else {
            other.gcd(self.modulo(&other))
        }
    }

    fn add(self, other: Self) -> Self {
        self ^ other
    }

    fn mulmod(self, other: Self, modulo: Self) -> Self {
        if self == 0 || other == 0 {
            return 0;
        }

        let mut res: Self = 0;
        let mut a = self;
        let mut b = other;

        if b & 1 > 0 {
            res = res.add(a).modulo(&modulo);
        }

        while b != 0 {
            a = (a << 1).modulo(&modulo);
            b >>= 1;
            if b & 1 > 0 {
                res = res.add(a).modulo(&modulo);
            }
        }

        res
    }
}

// qp computes the polynomial (x^(2^p)-x) mod g. This is needed for the
// reducibility test.
fn qp(p: i32, g: Polynom64) -> Polynom64 {
    // start with x
    let mut res: Polynom64 = 2;

    for _ in 0..p {
        // repeatedly square res
        res = res.mulmod(res, g);
    }

    // add x
    res.add(2).modulo(&g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::crypto::hasher::hash;
    use insta::assert_ron_snapshot;
    use rand::prelude::*;
    use std::io::{Cursor, repeat};

    pub(super) const POLY: Polynom64 = 0x003D_A335_8B4D_C173;
    /// The Splitmask is used to determine if a chunk is a chunk boundary.
    const SIZE: usize = 1 << 20;
    /// The size of a megabyte.
    const MB: usize = 1024 * constants::KB;
    /// The minimum size of a chunk.
    const MIN_SIZE: usize = 512 * constants::KB;
    /// The maximum size of a chunk.
    const MAX_SIZE: usize = 8 * MB;

    pub(super) fn random_data(seed: u64, len: usize) -> Vec<u8> {
        let mut data = vec![0; len];
        StdRng::seed_from_u64(seed).fill_bytes(&mut data);
        data
    }

    fn chunker<R: Read + Send>(reader: R, size_hint: usize) -> ChunkIter<R> {
        ChunkIter::new(POLY, SIZE, MIN_SIZE, MAX_SIZE, reader, size_hint).unwrap()
    }

    #[test]
    fn chunk_random() {
        const RANDOM_DATA_SIZE: usize = 32 * 1024 * 1024;
        const SEED: u64 = 23;
        let mut reader = Cursor::new(random_data(SEED, RANDOM_DATA_SIZE));
        let chunker = chunker(&mut reader, 0);
        let chunks: Vec<_> = chunker
            .map(|chunk| {
                let chunk = chunk.unwrap();
                (chunk.len(), hash(&chunk))
            })
            .collect();

        assert_ron_snapshot!(chunks);
    }

    #[test]
    fn min_size_below_the_read_buffer_keeps_chunks_within_max_size() {
        const SMALL_SIZE: usize = 8 * constants::KB;
        const SMALL_MIN_SIZE: usize = 2 * constants::KB;
        const SMALL_MAX_SIZE: usize = 32 * constants::KB;
        let data = random_data(23, MB);
        let chunks = ChunkIter::new(
            POLY,
            SMALL_SIZE,
            SMALL_MIN_SIZE,
            SMALL_MAX_SIZE,
            Cursor::new(&data),
            0,
        )
        .unwrap()
        .collect::<RusticResult<Vec<_>>>()
        .unwrap();

        assert!(chunks.iter().all(|chunk| chunk.len() <= SMALL_MAX_SIZE));
        assert_eq!(chunks.concat(), data);
    }

    #[test]
    fn min_size_has_to_fill_the_rolling_hash_window() {
        let window = constants::WINDOW_SIZE;

        assert!(check_rabin_params(SIZE, window - 1, MAX_SIZE).is_err());
        assert!(check_rabin_params(SIZE, window, MAX_SIZE).is_ok());
    }

    #[test]
    fn zeros_filling_the_window_right_after_min_size_cut_there() {
        const ZEROS_START: usize = MIN_SIZE - 50;
        let mut data = random_data(23, MIN_SIZE + 1000);
        data.iter_mut()
            .filter(|byte| **byte == 0)
            .for_each(|byte| *byte = 1);
        data[ZEROS_START..ZEROS_START + 400].fill(0);

        let first_chunk = chunker(Cursor::new(data), 0).next().unwrap().unwrap();

        assert_eq!(first_chunk.len(), ZEROS_START + 64);
    }

    #[test]
    fn chunk_empty() {
        let empty: Vec<u8> = vec![];
        let mut reader = Cursor::new(empty);
        let chunker = chunker(&mut reader, 0);

        assert_eq!(0, chunker.into_iter().count());
    }

    #[test]
    fn chunk_empty_wrong_hint() {
        let empty: Vec<u8> = vec![];
        let mut reader = Cursor::new(empty);
        let chunker = chunker(&mut reader, 100);

        assert_eq!(0, chunker.into_iter().count());
    }

    #[test]
    fn chunk_zeros() {
        let mut reader = repeat(0u8);
        let mut chunker = chunker(&mut reader, usize::MAX);

        let chunk = chunker.next().unwrap().unwrap();
        assert_eq!(MIN_SIZE, chunk.len());
    }
}
