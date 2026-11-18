//! Rabin fingerprints of [`LANES`] windows at once, using byte shuffles as table lookups.
//!
//! Fingerprints are linear over GF(2), so a lookup in a 256-entry table of `u64`s equals the xor of lookups in two
//! 16-entry tables indexed by the low and high nibble of the byte. Each 16-entry table is in turn split into one
//! table per byte of the `u64` ("plane"), which is exactly what a 16-byte shuffle can look up. The fingerprints
//! of all lanes are kept in the same layout: one vector per byte of the fingerprint, holding that byte of every
//! lane. Shifting a fingerprint left by 8 is then just a matter of renumbering the planes.

// Helpers only get compiled with the SIMD level's target features if they are inlined into the `#[simd]` function.
#![allow(clippy::inline_always)]

use fearless_simd::{Level, Simd, SimdBase, SimdMask, dispatch, u8x16};
use fearless_simd_macros::simd;

use super::{super::constants::WINDOW_SIZE, split_lanes};

/// Number of windows hashed at once.
pub(super) const LANES: usize = 16;
/// Lanes are loaded and transposed in blocks of this many bytes.
pub(super) const LANE_BLOCK: usize = 16;
// `load_transposed` transposes a square block of `u8x16`s.
const _: () = assert!(LANES == 16 && LANE_BLOCK == 16);
/// Number of blocks a window spans.
const WINDOW_BLOCKS: usize = WINDOW_SIZE / LANE_BLOCK;
const _: () = assert!(WINDOW_SIZE.is_multiple_of(LANE_BLOCK));

/// Degree of the modulo polynomial these tables support, which is the one rustic and restic generate.
const DEGREE: u32 = 53;
/// Number of bytes of a fingerprint that can be non-zero.
const PLANES: usize = DEGREE.div_ceil(8) as usize;
/// The plane holding the lowest bits of the fingerprint's top byte, which indexes the modulo table.
const TOP_PLANE: usize = (DEGREE as usize - 8) / 8;
/// Position of the top byte's lowest bit within [`TOP_PLANE`].
const TOP_SHIFT: u32 = (DEGREE - 8) % 8;
/// Number of fingerprint bytes compared against the split mask, limiting the split mask to 32 bits.
const MASK_PLANES: usize = 4;

type NibbleTables = [[u8; 16]; PLANES];

/// Returns whether `level` can shuffle bytes natively, which the sliced fingerprints rely on to be fast.
pub(super) fn has_byte_shuffle(level: Level) -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        level.as_neon().is_some()
    }
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        level.as_sse4_2().is_some()
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        _ = level;
        false
    }
}

/// Finds cut points like [`super::CutFinder`], but hashes [`LANES`] windows at once using SIMD.
///
/// Holds the Rabin tables of a [`super::CutFinder`], split into nibble-indexed byte planes.
#[derive(Debug, Clone)]
pub(super) struct SlicedFinder {
    level: Level,
    mod_lo: NibbleTables,
    mod_hi: NibbleTables,
    leave_lo: NibbleTables,
    leave_hi: NibbleTables,
    split_mask: [u8; MASK_PLANES],
}

impl SlicedFinder {
    /// Creates a `SlicedFinder` from the tables of a [`super::CutFinder`].
    ///
    /// This is only faster than the scalar `CutFinder` if `level` passes [`has_byte_shuffle`].
    ///
    /// Returns `None` if the polynomial's degree isn't [`DEGREE`] or `split_mask` doesn't fit in 32 bits.
    pub(super) fn new(
        level: Level,
        mod_table: &[u64; 256],
        leave_table: &[u64; 256],
        degree: u32,
        split_mask: u64,
    ) -> Option<Self> {
        if degree != DEGREE {
            return None;
        }
        let split_mask = u32::try_from(split_mask).ok()?;
        let (mod_lo, mod_hi) = split_table(mod_table);
        let (leave_lo, leave_hi) = split_table(leave_table);
        Some(Self {
            level,
            mod_lo,
            mod_hi,
            leave_lo,
            leave_hi,
            split_mask: split_mask.to_le_bytes(),
        })
    }

    /// Splits `start..start + LANES * lane_len` into [`LANES`] consecutive lanes, hashes them in lockstep, and
    /// returns the first cut point.
    ///
    /// `lane_len` must be a multiple of [`LANE_BLOCK`] and `start` must be at least [`WINDOW_SIZE`].
    pub(super) fn find_cut_in_lanes(
        &self,
        data: &[u8],
        start: usize,
        lane_len: usize,
    ) -> Option<usize> {
        dispatch!(self.level, simd => find_cut_in_lanes(simd, self, data, start, lane_len))
    }
}

#[allow(clippy::cast_possible_truncation)]
fn split_table(table: &[u64; 256]) -> (NibbleTables, NibbleTables) {
    let mut lo = [[0; 16]; PLANES];
    let mut hi = [[0; 16]; PLANES];
    for nibble in 0..16 {
        for plane in 0..PLANES {
            lo[plane][nibble] = (table[nibble] >> (8 * plane)) as u8;
            hi[plane][nibble] = (table[nibble << 4] >> (8 * plane)) as u8;
        }
    }
    (lo, hi)
}

struct Tables<S: Simd> {
    mod_lo: [u8x16<S>; PLANES],
    mod_hi: [u8x16<S>; PLANES],
    leave_lo: [u8x16<S>; PLANES],
    leave_hi: [u8x16<S>; PLANES],
    split_mask: [u8x16<S>; MASK_PLANES],
}

impl<S: Simd> Tables<S> {
    #[inline(always)]
    fn load(simd: S, finder: &SlicedFinder) -> Self {
        let SlicedFinder {
            level: _,
            mod_lo,
            mod_hi,
            leave_lo,
            leave_hi,
            split_mask,
        } = finder;
        let load = |table: &NibbleTables| table.map(|plane| u8x16::load_array(simd, plane));
        Self {
            mod_lo: load(mod_lo),
            mod_hi: load(mod_hi),
            leave_lo: load(leave_lo),
            leave_hi: load(leave_hi),
            split_mask: split_mask.map(|byte| u8x16::splat(simd, byte)),
        }
    }

    #[inline(always)]
    fn slide(
        &self,
        planes: [u8x16<S>; PLANES],
        leaving: u8x16<S>,
        entering: u8x16<S>,
    ) -> [u8x16<S>; PLANES] {
        let leaving_lo = leaving & 0x0f;
        let leaving_hi = leaving >> 4;
        let mut data = shift_in(planes, entering);
        for plane in 0..PLANES {
            data[plane] ^= self.leave_lo[plane].swizzle_dyn(leaving_lo)
                ^ self.leave_hi[plane].swizzle_dyn(leaving_hi);
        }
        self.reduce(planes, data)
    }

    /// Like [`Self::slide`], but for a window that isn't full yet, so no byte leaves it.
    #[inline(always)]
    fn append(&self, planes: [u8x16<S>; PLANES], entering: u8x16<S>) -> [u8x16<S>; PLANES] {
        self.reduce(planes, shift_in(planes, entering))
    }

    /// Xors the modulo of the top byte of `planes`, which is shifted out of the fingerprints, into `data`.
    ///
    /// Takes `data` with everything else already xored in, so that this lookup is the only one on the dependency
    /// chain between consecutive fingerprints.
    #[inline(always)]
    fn reduce(
        &self,
        planes: [u8x16<S>; PLANES],
        mut data: [u8x16<S>; PLANES],
    ) -> [u8x16<S>; PLANES] {
        let top = (planes[TOP_PLANE] >> TOP_SHIFT) | (planes[TOP_PLANE + 1] << (8 - TOP_SHIFT));
        let top_lo = top & 0x0f;
        let top_hi = top >> 4;
        for plane in 0..PLANES {
            data[plane] ^=
                self.mod_lo[plane].swizzle_dyn(top_lo) ^ self.mod_hi[plane].swizzle_dyn(top_hi);
        }
        data
    }

    /// Returns the bits of the fingerprints selected by the split mask, which are zero for cut points.
    #[inline(always)]
    fn masked(&self, planes: &[u8x16<S>; PLANES]) -> u8x16<S> {
        let mut masked = planes[0] & self.split_mask[0];
        for plane in 1..MASK_PLANES {
            masked |= planes[plane] & self.split_mask[plane];
        }
        masked
    }
}

/// Shifts the fingerprints left by 8 bits, dropping the top plane and shifting `entering` in at the bottom.
#[inline(always)]
fn shift_in<S: Simd>(planes: [u8x16<S>; PLANES], entering: u8x16<S>) -> [u8x16<S>; PLANES] {
    let mut shifted = [entering; PLANES];
    shifted[1..].copy_from_slice(&planes[..PLANES - 1]);
    shifted
}

/// Loads a block of every lane and transposes them, so that vector `i` holds byte `i` of every lane.
#[inline(always)]
fn load_transposed<S: Simd>(
    simd: S,
    blocks: [&[u8; LANE_BLOCK]; LANES],
) -> [u8x16<S>; LANE_BLOCK] {
    let mut rows = blocks.map(|block| u8x16::load_array_ref(simd, block));
    // Interleaving rows `j` and `j + 8` four times is a perfect shuffle of the row and column bits, i.e. a
    // transpose.
    for _ in 0..4 {
        let mut next = rows;
        for row in 0..8 {
            next[2 * row] = rows[row].zip_low(rows[row + 8]);
            next[2 * row + 1] = rows[row].zip_high(rows[row + 8]);
        }
        rows = next;
    }
    rows
}

#[simd]
#[allow(clippy::needless_range_loop)]
fn find_cut_in_lanes<S: Simd>(
    simd: S,
    finder: &SlicedFinder,
    data: &[u8],
    start: usize,
    lane_len: usize,
) -> Option<usize> {
    let tables = Tables::load(simd, finder);
    let zero = u8x16::splat(simd, 0);

    let lanes = split_lanes::<LANES, LANE_BLOCK>(data, start, lane_len);

    let mut window = [[zero; LANE_BLOCK]; WINDOW_BLOCKS];
    let mut planes = [zero; PLANES];
    for block in 0..WINDOW_BLOCKS {
        let entering = load_transposed(simd, lanes.map(|lane| &lane[block]));
        for byte in entering {
            planes = tables.append(planes, byte);
        }
        window[block] = entering;
    }

    let mut found = 0_u64;
    let mut cuts = [0; LANES];
    for block in WINDOW_BLOCKS..lanes[0].len() {
        let entering = load_transposed(simd, lanes.map(|lane| &lane[block]));
        let leaving = std::mem::replace(&mut window[block % WINDOW_BLOCKS], entering);
        let mut masked = [zero; LANE_BLOCK];
        // Cut points are rare, so rather than checking every step for one, only check whether a block has any.
        let mut least = u8x16::splat(simd, u8::MAX);
        for i in 0..LANE_BLOCK {
            planes = tables.slide(planes, leaving[i], entering[i]);
            masked[i] = tables.masked(&planes);
            least = least.min(masked[i]);
        }
        if !least.simd_eq(0).any_true() {
            continue;
        }

        let offset = (block - WINDOW_BLOCKS) * LANE_BLOCK;
        for i in 0..LANE_BLOCK {
            let new = masked[i].simd_eq(0).to_bitmask() & !found;
            for lane in 0..LANES {
                if new & (1 << lane) != 0 {
                    cuts[lane] = start + lane * lane_len + offset + i;
                }
            }
            found |= new;
            if found & 1 != 0 {
                return Some(cuts[0]);
            }
        }
    }

    if found == 0 {
        return None;
    }
    Some(cuts[found.trailing_zeros() as usize])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::rabin::{
        scan::CutFinder,
        tests::{POLY, random_data},
    };

    /// Runs the SIMD code on `level`, even if that level can't shuffle bytes natively.
    fn find_cut_on_level(
        level: Level,
        split_mask: u64,
        data: &[u8],
        start: usize,
        lane_len: usize,
    ) -> Option<usize> {
        let scalar = CutFinder::with_level(POLY, split_mask, Level::fallback()).unwrap();
        SlicedFinder::new(level, &scalar.mod_table, &scalar.leave_table, DEGREE, split_mask)
            .unwrap()
            .find_cut_in_lanes(data, start, lane_len)
    }

    #[test]
    fn find_cut_in_lanes_matches_scalar_on_every_level() {
        let data = random_data(5, 40_000);
        let lane_len = 1024;

        for mask_bits in [6, 10, 14, 18] {
            let split_mask = (1 << mask_bits) - 1;
            let scalar = CutFinder::with_level(POLY, split_mask, Level::fallback()).unwrap();
            for start in [WINDOW_SIZE, 1000, 2001, 5555] {
                let range = start..start + LANES * lane_len;
                let expected = scalar.find_cut(&data, range);
                for level in [Level::new(), Level::fallback()] {
                    assert_eq!(
                        find_cut_on_level(level, split_mask, &data, start, lane_len),
                        expected,
                        "mask_bits = {mask_bits}, start = {start}, level = {level:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn unsupported_parameters_are_rejected() {
        let scalar = CutFinder::with_level(POLY, 0, Level::fallback()).unwrap();
        let finder = |degree, split_mask| {
            SlicedFinder::new(
                Level::fallback(),
                &scalar.mod_table,
                &scalar.leave_table,
                degree,
                split_mask,
            )
        };

        assert!(finder(DEGREE, u64::from(u32::MAX)).is_some());
        assert!(finder(DEGREE, 1 << 32).is_none());
        assert!(finder(DEGREE - 1, 0).is_none());
        assert!(!has_byte_shuffle(Level::fallback()));
    }
}
