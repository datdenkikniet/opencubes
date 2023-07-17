#[cfg(test)]
mod test;

use std::{
    collections::HashSet,
    io::{Read, Write},
};

use parking_lot::RwLock;

mod iterator;

mod from_file;
pub use from_file::{Compression, PolyCubeFile};

pub fn make_bar(len: u64) -> indicatif::ProgressBar {
    use indicatif::{ProgressBar, ProgressStyle};

    let bar = ProgressBar::new(len);

    let pos_width = format!("{len}").len();

    let template =
        format!("[{{elapsed_precise}}] {{bar:40.cyan/blue}} {{pos:>{pos_width}}}/{{len}} {{msg}}");

    bar.set_style(
        ProgressStyle::with_template(&template)
            .unwrap()
            .progress_chars("#>-"),
    );
    bar
}

/// A polycube
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PolyCube {
    dim_1: u8,
    dim_2: u8,
    dim_3: u8,
    filled: Vec<bool>,
}

impl core::fmt::Display for PolyCube {
    // Format the polycube in a somewhat more easy to digest
    // format.
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut xy = String::new();

        for _ in 0..self.dim_3 {
            xy.push('-');
        }
        xy.push('\n');

        for x in 0..self.dim_1 {
            for y in 0..self.dim_2 {
                for z in 0..self.dim_3 {
                    if self.is_set(x, y, z) {
                        xy.push('1');
                    } else {
                        xy.push('0');
                    }
                }
                xy.push('\n');
            }

            for _ in 0..self.dim_3 {
                xy.push('-');
            }
            xy.push('\n');
        }

        write!(f, "{}", xy.trim_end())
    }
}

/// Creating a new polycube from a triple-nested vector
/// is convenient if/when you're writing them out
/// by hand.
impl From<Vec<Vec<Vec<bool>>>> for PolyCube {
    fn from(value: Vec<Vec<Vec<bool>>>) -> Self {
        let dim_1 = value.len() as u8;
        let dim_2 = value[0].len() as u8;
        let dim_3 = value[0][0].len() as u8;

        let mut poly_cube = PolyCube::new(dim_1, dim_2, dim_3);

        for d3 in 0..poly_cube.dim_3 {
            for d2 in 0..poly_cube.dim_2 {
                for d1 in 0..poly_cube.dim_1 {
                    poly_cube
                        .set_to(d1, d2, d3, value[d1 as usize][d2 as usize][d3 as usize])
                        .unwrap();
                }
            }
        }

        poly_cube
    }
}

impl PolyCube {
    /// Get the dimensions of this polycube
    pub fn dims(&self) -> (u8, u8, u8) {
        (self.dim_1, self.dim_2, self.dim_3)
    }

    pub fn present_cubes(&self) -> usize {
        self.filled.iter().filter(|v| **v).count()
    }

    pub fn unpack(mut from: impl Read) -> std::io::Result<Self> {
        let mut xyz = [0u8; 3];
        from.read_exact(&mut xyz)?;

        let [d1, d2, d3] = xyz;
        let [d1, d2, d3] = [d1, d2, d3];

        let size = d1 as usize * d2 as usize * d3 as usize;
        let mut data = vec![0u8; (size + 7) / 8];
        from.read_exact(&mut data)?;

        let mut filled = Vec::with_capacity(size);

        data.iter().for_each(|v| {
            for s in 0..8 {
                let is_set = ((*v >> s) & 0x1) == 0x1;
                if filled.capacity() != filled.len() {
                    filled.push(is_set);
                }
            }
        });

        Ok(Self::new_raw(d1, d2, d3, filled))
    }

    pub fn pack(&self, mut write: impl Write) -> std::io::Result<()> {
        let len = self.dim_1 as usize * self.dim_2 as usize * self.dim_3 as usize;
        let byte_len = (len + 7) / 8;

        let mut out_bytes = vec![0u8; byte_len + 3];
        out_bytes[0] = self.dim_1;
        out_bytes[1] = self.dim_2;
        out_bytes[2] = self.dim_3;

        let mut filled = self.filled.iter();
        out_bytes.iter_mut().skip(3).for_each(|v| {
            for s in 0..8 {
                if let Some(true) = filled.next() {
                    *v |= 1 << s;
                }
            }
        });

        write.write_all(&out_bytes)?;

        Ok(())
    }

    /// Find the ordering between two rotated versions of the same
    /// PolyCube.
    ///
    /// This function only produces valid results if `self` and `other` are
    /// two different rotations of the same PolyCube.
    pub fn canonical_ordering(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        macro_rules! check_next {
            ($name:ident) => {
                match self.$name.cmp(&other.$name) {
                    Ordering::Equal => {}
                    ord => return ord,
                }
            };
        }

        check_next!(dim_1);
        check_next!(dim_2);
        check_next!(dim_3);

        // I don't think this does what I expect it to do...
        self.filled.cmp(&other.filled)
    }

    /// Calculate the offset into `self.filled` using the provided offsets
    /// within each dimension.
    fn offset(&self, dim_1: u8, dim_2: u8, dim_3: u8) -> Option<usize> {
        if dim_1 < self.dim_1 && dim_2 < self.dim_2 && dim_3 < self.dim_3 {
            let d1 = dim_1 as usize * self.dim_2 as usize * self.dim_3 as usize;
            let d2 = dim_2 as usize * self.dim_3 as usize;
            let d3 = dim_3 as usize;
            let index = d1 + d2 + d3;

            Some(index)
        } else {
            None
        }
    }

    pub fn new_raw(dim_1: u8, dim_2: u8, dim_3: u8, filled: Vec<bool>) -> Self {
        Self {
            dim_1,
            dim_2,
            dim_3,

            filled,
        }
    }

    /// Create a new [`PolyCube`] with dimensions `(dim_1, dim_2, dim_3)` and
    /// a new allocation tracker.
    pub fn new(dim_1: u8, dim_2: u8, dim_3: u8) -> Self {
        let filled = (0..dim_1 as usize * dim_2 as usize * dim_3 as usize)
            .map(|_| false)
            .collect();

        Self {
            dim_1,
            dim_2,
            dim_3,
            filled,
        }
    }

    /// Create a new [`PolyCube`] with dimensions `(side, side, side)`, and
    /// a new allocation tracker.
    pub fn new_equal_sides(side: u8) -> Self {
        Self::new(side, side, side)
    }

    /// Set the state of the box located at `(d1, d2, d3)` to `set`.
    pub fn set_to(&mut self, d1: u8, d2: u8, d3: u8, set: bool) -> Result<(), ()> {
        let idx = self.offset(d1, d2, d3).ok_or(())?;
        self.filled[idx] = set;
        Ok(())
    }

    /// Set the box located at `(d1, d2, d3)` to be filled.
    pub fn set(&mut self, d1: u8, d2: u8, d3: u8) -> Result<(), ()> {
        self.set_to(d1, d2, d3, true)
    }

    /// Returns whether the box located at `(d1, d2, d3)` is filled.
    pub fn is_set(&self, d1: u8, d2: u8, d3: u8) -> bool {
        self.offset(d1, d2, d3)
            .map(|v| self.filled[v])
            .unwrap_or(false)
    }

    /// Create a new [`PolyCube`], representing `self` rotated `k` times in the plane indicated by `a1` and `a2`.
    pub fn rot90(self, k: usize, (a1, a2): (usize, usize)) -> PolyCube {
        assert!(a1 <= 2, "a1 must be <= 2");
        assert!(a2 <= 2, "a2 must be <= 2");

        let k = k % 4;

        if k == 0 {
            return self;
        }

        if k == 2 {
            return self.flip(a1).flip(a2);
        }

        let mut axes: [usize; 3] = [0, 1, 2];
        let saved = axes[a1];
        axes[a1] = axes[a2];
        axes[a2] = saved;

        if k == 1 {
            self.flip(a2).transpose(axes[0], axes[1], axes[2])
        } else {
            // k == 3
            self.transpose(axes[0], axes[1], axes[2]).flip(a2)
        }
    }

    /// Create a new [`PolyCube`], representing `self` transposed according to `a1`, `a2`, and `a3`.
    ///
    /// The axes of the returned [`PolyCube`] will be those of `self`, rearranged according to the
    /// provided axes.
    pub fn transpose(&self, a1: usize, a2: usize, a3: usize) -> PolyCube {
        assert!(a1 != a2);
        assert!(a1 != a3);
        assert!(a2 != a3);
        assert!(a1 <= 2);
        assert!(a2 <= 2);
        assert!(a3 <= 2);

        let original_dimension = [self.dim_1, self.dim_2, self.dim_3];
        let [td1, td2, td3] = [
            original_dimension[a1],
            original_dimension[a2],
            original_dimension[a3],
        ];

        let mut new_cube = PolyCube::new(td1, td2, td3);

        for d1 in 0..self.dim_1 {
            for d2 in 0..self.dim_2 {
                for d3 in 0..self.dim_3 {
                    let original = [d1, d2, d3];
                    let [t1, t2, t3] = [original[a1], original[a2], original[a3]];

                    let orig_idx = self.offset(d1, d2, d3).unwrap();
                    let transposed_idx = new_cube.offset(t1, t2, t3).unwrap();

                    new_cube.filled[transposed_idx] = self.filled[orig_idx];
                }
            }
        }

        new_cube
    }

    /// Create a new [`PolyCube`], representing `self` flipped along `axis`.
    pub fn flip(&self, axis: usize) -> PolyCube {
        assert!(axis <= 2, "Axis must be <= 2");

        let mut new_cube = PolyCube::new(self.dim_1, self.dim_2, self.dim_3);

        macro_rules! flip {
            ($flipped_idx:expr) => {
                for d1 in 0..self.dim_1 {
                    for d2 in 0..self.dim_2 {
                        for d3 in 0..self.dim_3 {
                            let idx_1 = self.offset(d1, d2, d3).unwrap();
                            let idx_2 = $flipped_idx(d1, d2, d3).unwrap();

                            new_cube.filled[idx_2] = self.filled[idx_1];
                        }
                    }
                }
            };
        }

        match axis {
            0 => flip!(|d1, d2, d3| self.offset(self.dim_1 - d1 - 1, d2, d3)),
            1 => flip!(|d1, d2, d3| self.offset(d1, self.dim_2 - d2 - 1, d3)),
            2 => flip!(|d1, d2, d3| self.offset(d1, d2, self.dim_3 - d3 - 1)),
            _ => unreachable!(),
        }

        new_cube
    }

    /// Create a new [`PolyCube`] that has an extra box-space on all sides
    /// of the polycube.
    pub fn pad_one(&self) -> PolyCube {
        let mut cube_next = PolyCube::new(self.dim_1 + 2, self.dim_2 + 2, self.dim_3 + 2);

        for d1 in 0..self.dim_1 {
            for d2 in 0..self.dim_2 {
                for d3 in 0..self.dim_3 {
                    cube_next
                        .set_to(d1 + 1, d2 + 1, d3 + 1, self.is_set(d1, d2, d3))
                        .unwrap();
                }
            }
        }

        cube_next
    }

    /// Obtain a list of [`PolyCube`]s representing all unique expansions of the
    /// items in `from_set`.
    ///
    /// If the feature `indicatif` is enabled, this also prints a progress bar.
    pub fn unique_expansions<'a, I>(use_bar: bool, n: usize, from_set: I) -> Vec<PolyCube>
    where
        I: Iterator<Item = &'a PolyCube> + ExactSizeIterator,
    {
        let bar = make_bar(from_set.len() as u64);

        let mut this_level = HashSet::new();

        let mut iter = 0;
        for value in from_set {
            iter += 1;
            for expansion in value.expand().map(|v| v.crop()) {
                let max = expansion
                    .all_rotations()
                    .max_by(Self::canonical_ordering)
                    .unwrap();
                let missing = !this_level.contains(&max);

                if missing {
                    this_level.insert(max);
                }
            }

            if use_bar {
                bar.inc(1);

                // Try to avoid doing this too often
                if iter % (this_level.len() / 100).max(100) == 0 {
                    let len = this_level.len();
                    bar.set_message(format!("Unique polycubes for N = {n} so far: {len}"));
                }
            }
        }

        if use_bar {
            let len = this_level.len();
            bar.set_message(format!("Unique polycubes for N = {n}: {len}"));
            bar.finish();
        }

        this_level.into_iter().collect()
    }

    /// Check whether this cube is already cropped.
    pub fn is_cropped(&self) -> bool {
        macro_rules! direction {
            ($d1:expr, $d2:expr, $d3:expr, $pred:expr) => {{
                for d1 in $d1 {
                    let mut has_nonzero = false;
                    for d2 in 0..$d2 {
                        for d3 in 0..$d3 {
                            has_nonzero |= $pred(d1, d2, d3);
                            if has_nonzero {
                                break;
                            }
                        }
                    }

                    if !has_nonzero {
                        return false;
                    } else {
                        break;
                    }
                }
            }};
        }

        let d1_first = |d1, d2, d3| self.is_set(d1, d2, d3);
        direction!(0..self.dim_1, self.dim_2, self.dim_3, d1_first);
        direction!((0..self.dim_1).rev(), self.dim_2, self.dim_3, d1_first);

        let d2_first = |d2, d1, d3| self.is_set(d1, d2, d3);
        direction!(0..self.dim_2, self.dim_1, self.dim_3, d2_first);
        direction!((0..self.dim_2).rev(), self.dim_1, self.dim_3, d2_first);

        let d3_first = |d3, d1, d2| self.is_set(d1, d2, d3);
        direction!(0..self.dim_3, self.dim_1, self.dim_2, d3_first);
        direction!((0..self.dim_3).rev(), self.dim_1, self.dim_2, d3_first);

        return true;
    }

    /// Create a new [`PolyCube`] representing `self` but cropped.
    ///
    /// Cropping means that there are no planes without any present boxes.
    pub fn crop(&self) -> PolyCube {
        macro_rules! direction {
            ($d1:expr, $d2:expr, $d3:expr, $pred:expr) => {{
                let mut all_zero_count: u8 = 0;

                for d1 in $d1 {
                    let mut has_nonzero = false;
                    for d2 in 0..$d2 {
                        for d3 in 0..$d3 {
                            has_nonzero |= $pred(d1, d2, d3);
                            if has_nonzero {
                                break;
                            }
                        }
                    }

                    if !has_nonzero {
                        all_zero_count += 1;
                    } else {
                        break;
                    }
                }

                all_zero_count
            }};
        }

        let d1_first = |d1, d2, d3| self.is_set(d1, d2, d3);
        let d1_left = direction!(0..self.dim_1, self.dim_2, self.dim_3, d1_first);

        // If there are `dim_1` planes to be removed, we have to remove them all,
        // which means that there are no boxes present in this polycube, at all.
        if d1_left == self.dim_1 {
            return PolyCube {
                // NOTE: this doesn't increase allocation count, since
                // Vec::new() does not allocate for size 0.
                dim_1: 0,
                dim_2: 0,
                dim_3: 0,
                filled: Vec::new(),
            };
        }

        let d1_right = direction!((0..self.dim_1).rev(), self.dim_2, self.dim_3, d1_first);

        let d2_first = |d2, d1, d3| self.is_set(d1, d2, d3);
        let d2_left = direction!(0..self.dim_2, self.dim_1, self.dim_3, d2_first);
        let d2_right = direction!((0..self.dim_2).rev(), self.dim_1, self.dim_3, d2_first);

        let d3_first = |d3, d1, d2| self.is_set(d1, d2, d3);
        let d3_left = direction!(0..self.dim_3, self.dim_1, self.dim_2, d3_first);
        let d3_right = direction!((0..self.dim_3).rev(), self.dim_1, self.dim_2, d3_first);

        let mut new_cube = PolyCube::new(
            self.dim_1 - d1_left - d1_right,
            self.dim_2 - d2_left - d2_right,
            self.dim_3 - d3_left - d3_right,
        );

        for d1 in 0..new_cube.dim_1 {
            for d2 in 0..new_cube.dim_2 {
                for d3 in 0..new_cube.dim_3 {
                    let d1_from = d1 + d1_left;
                    let d2_from = d2 + d2_left;
                    let d3_from = d3 + d3_left;

                    let is_set = self.is_set(d1_from, d2_from, d3_from);
                    new_cube.set_to(d1, d2, d3, is_set).unwrap();
                }
            }
        }

        new_cube
    }
}

impl PolyCube {
    pub fn unique_expansions_rayon<'a, I>(use_bar: bool, n: usize, from_set: I) -> Vec<PolyCube>
    where
        I: Iterator<Item = &'a PolyCube> + ExactSizeIterator + Clone + Send + Sync,
    {
        use rayon::prelude::*;

        if from_set.len() == 0 {
            return Vec::new();
        }

        let available_parallelism = num_cpus::get();

        let chunk_size = (from_set.len() / available_parallelism) + 1;
        let chunks = (from_set.len() + chunk_size - 1) / chunk_size;

        let bar = make_bar(from_set.len() as u64);

        let chunk_iterator = (0..chunks).into_par_iter().map(|v| {
            from_set
                .clone()
                .skip(v * chunk_size)
                .take(chunk_size)
                .into_iter()
        });

        let this_level = RwLock::new(HashSet::new());

        chunk_iterator.for_each(|v| {
            for value in v {
                for expansion in value.expand().map(|v| v.crop()) {
                    let max = expansion
                        .all_rotations()
                        .max_by(Self::canonical_ordering)
                        .unwrap();

                    let missing = !this_level.read().contains(&max);

                    if missing {
                        this_level.write().insert(max);
                    }
                }

                if use_bar {
                    bar.inc(1);

                    // Try to avoid doing this too often
                    if bar.position() % (this_level.read().len() as u64 / 100).max(100) == 0 {
                        let len = this_level.read().len();
                        bar.set_message(format!("Unique polycubes for N = {n} so far: {len}",));
                    }
                }
            }
        });

        if use_bar {
            let len = this_level.read().len();
            bar.set_message(format!("Unique polycubes for N = {n}: {len}",));
            bar.finish();
        }

        this_level.into_inner().into_iter().collect()
    }
}
