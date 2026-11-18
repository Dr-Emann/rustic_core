use std::ops::Range;

use rustic_cdc::{Polynom, Polynom64, Rabin64};

use super::constants::WINDOW_SIZE;

/// Finds chunk boundaries by computing the Rabin fingerprint of [`WINDOW_SIZE`]-byte windows.
///
/// The fingerprints are identical to those of [`Rabin64`] with a 64-byte window. Unlike [`Rabin64`], the window
/// is not copied into a ring buffer: the byte leaving the window is read back from the scanned data.
#[derive(Debug, Clone)]
pub(super) struct CutFinder {
    mod_table: [u64; 256],
    /// Contribution of the byte leaving the window to the next fingerprint.
    ///
    /// Removing `out_table[leaving]` and then sliding in a byte is linear over GF(2), so the removal can be
    /// folded into a single table that is xored in after the shift. This keeps the lookup off the dependency
    /// chain between consecutive fingerprints.
    leave_table: [u64; 256],
    shift: u32,
    split_mask: u64,
}

impl CutFinder {
    /// Creates a `CutFinder` for the modulo polynomial `poly`.
    ///
    /// A window ending at some position is a cut point if its fingerprint has no bits of `split_mask` set.
    /// Returns `None` if `poly` has a degree below 8.
    pub(super) fn new(poly: Polynom64, split_mask: u64) -> Option<Self> {
        let shift = u32::try_from(poly.degree() - 8).ok()?;
        let out_table = Rabin64::calculate_out_table(WINDOW_SIZE, &poly);
        let mod_table = Rabin64::calculate_mod_table(&poly);
        let mut leave_table = [0; 256];
        for (entry, out) in leave_table.iter_mut().zip(out_table) {
            *entry = (out << 8) ^ mod_table[top_byte(out, shift)];
        }
        Some(Self {
            mod_table,
            leave_table,
            shift,
            split_mask,
        })
    }

    /// Returns the first position in `range` at which the window ending at (and including) that position is a
    /// cut point.
    ///
    /// # Panics
    ///
    /// * If `range` is non-empty and the window ending at `range.start` doesn't fit in `data`.
    pub(super) fn find_cut(&self, data: &[u8], range: Range<usize>) -> Option<usize> {
        let Range { start, end } = range;
        if start >= end {
            return None;
        }

        let mut hash = self.hash_window(&data[start + 1 - WINDOW_SIZE..=start]);
        if self.is_cut(hash) {
            return Some(start);
        }

        let entering = &data[start + 1..end];
        let leaving = &data[start + 1 - WINDOW_SIZE..end - WINDOW_SIZE];
        for (pos, (&entering, &leaving)) in (start + 1..).zip(entering.iter().zip(leaving)) {
            hash = self.slide(hash, leaving, entering);
            if self.is_cut(hash) {
                return Some(pos);
            }
        }
        None
    }

    fn hash_window(&self, window: &[u8]) -> u64 {
        let mut hash = 0;
        for &byte in window {
            hash = self.slide(hash, 0, byte);
        }
        hash
    }

    #[inline]
    fn slide(&self, hash: u64, leaving: u8, entering: u8) -> u64 {
        let data = u64::from(entering) ^ self.leave_table[usize::from(leaving)];
        ((hash << 8) ^ data) ^ self.mod_table[top_byte(hash, self.shift)]
    }

    #[inline]
    fn is_cut(&self, hash: u64) -> bool {
        hash & self.split_mask == 0
    }
}

#[inline]
#[allow(clippy::cast_possible_truncation)]
fn top_byte(hash: u64, shift: u32) -> usize {
    ((hash >> shift) & 0xff) as usize
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chunker::rabin::tests::{POLY, random_data};
    use proptest::prelude::*;
    use rustic_cdc::RollingHash64;

    fn reference_find_cut(split_mask: u64, data: &[u8], range: Range<usize>) -> Option<usize> {
        let mut rabin = Rabin64::new_with_polynom(6, &POLY);
        let Range { start, end } = range;
        if start >= end {
            return None;
        }
        _ = rabin
            .reset_and_prefill_window(&mut data[start + 1 - WINDOW_SIZE..start].iter().copied());
        rabin.slide(data[start]);
        let mut pos = start;
        loop {
            if rabin.hash & split_mask == 0 {
                return Some(pos);
            }
            pos += 1;
            if pos == end {
                return None;
            }
            rabin.slide(data[pos]);
        }
    }

    proptest! {
        #[test]
        fn find_cut_matches_rabin64(
            seed: u64,
            len in WINDOW_SIZE..40_000,
            mask_bits in 0u32..16,
            start_offset: usize,
            range_len: usize,
        ) {
            let data = random_data(seed, len);
            let split_mask = (1 << mask_bits) - 1;
            let start = WINDOW_SIZE - 1 + start_offset % (len - WINDOW_SIZE + 2);
            let end = start + range_len % (len - start + 1);
            let finder = CutFinder::new(POLY, split_mask).unwrap();

            prop_assert_eq!(
                finder.find_cut(&data, start..end),
                reference_find_cut(split_mask, &data, start..end)
            );
        }
    }

    #[test]
    fn find_cut_reports_every_cut_of_rabin64() {
        let data = random_data(7, 200_000);
        let split_mask = (1 << 8) - 1;
        let finder = CutFinder::new(POLY, split_mask).unwrap();

        let mut start = WINDOW_SIZE - 1;
        let mut cuts = 0;
        loop {
            let expected = reference_find_cut(split_mask, &data, start..data.len());
            assert_eq!(finder.find_cut(&data, start..data.len()), expected);
            let Some(cut) = expected else { break };
            cuts += 1;
            start = cut + 1;
        }
        assert!(cuts > 500);
    }
}
