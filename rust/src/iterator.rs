use crate::PolyCube;

struct PlaneIterator {
    count: usize,
    plane: (usize, usize),
    base: PolyCube,
}

impl Iterator for PlaneIterator {
    type Item = PolyCube;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count <= 3 {
            let out = self.base.clone().rot90(self.count, self.plane);
            self.count += 1;
            Some(out)
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let left = 4 - self.count;
        (left, Some(left))
    }
}

impl ExactSizeIterator for PlaneIterator {
    fn len(&self) -> usize {
        let (lower, upper) = self.size_hint();
        // Note: This assertion is overly defensive, but it checks the invariant
        // guaranteed by the trait. If this trait were rust-internal,
        // we could use debug_assert!; assert_eq! will check all Rust user
        // implementations too.
        assert_eq!(upper, Some(lower));
        lower
    }
}

impl PolyCube {
    /// Obtain an iterator yielding all rotations of `self` in `plane`.
    pub fn rotations_in_plane(
        self,
        plane: (usize, usize),
    ) -> impl Iterator<Item = PolyCube> + ExactSizeIterator {
        PlaneIterator {
            count: 0,
            plane,
            base: self,
        }
    }

    /// Obtain an iterator yielding all possible rotations of `self`
    pub fn all_rotations(&self) -> impl Iterator<Item = PolyCube> + '_ {
        const _0_1: (usize, usize) = (0, 1);
        const _1_2: (usize, usize) = (1, 2);
        const _0_2: (usize, usize) = (0, 2);

        let rots_in_native_plane = self.clone().rotations_in_plane(_1_2);

        #[rustfmt::skip]
        let rotation_settings = [
            (2, _0_2, _1_2),
            (1, _0_2, _0_1),
            (3, _0_2, _0_1),
            (1, _0_1, _0_2),
            (3, _0_1, _0_2),
        ];

        let all_others = rotation_settings
            .into_iter()
            .flat_map(move |(k, p, rots_in_p)| {
                self.clone().rot90(k, p).rotations_in_plane(rots_in_p)
            });

        struct AllRotationsIter<I>
        where
            I: Iterator<Item = PolyCube>,
        {
            inner: I,
            rotations_checked: usize,
        }

        impl<I> Iterator for AllRotationsIter<I>
        where
            I: Iterator<Item = PolyCube>,
        {
            type Item = PolyCube;

            fn next(&mut self) -> Option<Self::Item> {
                let next = self.inner.next()?;
                self.rotations_checked += 1;
                Some(next)
            }

            fn size_hint(&self) -> (usize, Option<usize>) {
                let left = 24 - self.rotations_checked;
                (left, Some(left))
            }
        }

        impl<I> ExactSizeIterator for AllRotationsIter<I> where I: Iterator<Item = PolyCube> {}

        AllRotationsIter {
            inner: rots_in_native_plane.chain(all_others),
            rotations_checked: 0,
        }
    }
}

type Sides = std::array::IntoIter<(u8, u8, u8), 6>;

#[derive(Clone)]
struct ToggleIterator {
    dim_1: u8,
    dim_2: u8,
    dim_3: u8,
    iterating_cube: Option<Sides>,
    padded_cube: PolyCube,
    done: bool,
}

impl ToggleIterator {
    /// Move to the next possibly-occupied box
    ///
    /// Returns `true` if we're done iterating
    fn go_to_next(&mut self) -> bool {
        if self.dim_1 == self.padded_cube.dim_1 - 1 && self.dim_2 == self.padded_cube.dim_2 - 1 {
            self.dim_1 = 1;
            self.dim_2 = 1;
            self.dim_3 += 1;
        } else if self.dim_1 == self.padded_cube.dim_1 - 1 {
            self.dim_1 = 1;
            self.dim_2 += 1;
        } else {
            self.dim_1 += 1;
        }

        self.dim_3 == self.padded_cube.dim_3 - 1
    }

    fn faces(d1: u8, d2: u8, d3: u8) -> std::array::IntoIter<(u8, u8, u8), 6> {
        [
            (d1 + 1, d2, d3),
            (d1 - 1, d2, d3),
            (d1, d2 + 1, d3),
            (d1, d2 - 1, d3),
            (d1, d2, d3 + 1),
            (d1, d2, d3 - 1),
        ]
        .into_iter()
    }
}

impl Iterator for ToggleIterator {
    type Item = PolyCube;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.done && self.iterating_cube.is_none() {
                return None;
            }

            if self.iterating_cube.is_none() {
                loop {
                    let (d1, d2, d3) = (self.dim_1, self.dim_2, self.dim_3);

                    if self.go_to_next() {
                        self.done = true;
                        return None;
                    }

                    if self.padded_cube.is_set(d1, d2, d3) {
                        self.iterating_cube = Some(Self::faces(d1, d2, d3));
                        break;
                    }
                }
            }

            if let Some(sides) = &mut self.iterating_cube {
                let (d1, d2, d3) = sides.next().unwrap();

                if sides.len() == 0 {
                    self.iterating_cube.take();
                }

                // If the cube is already set, skip this face
                if self.padded_cube.is_set(d1, d2, d3) {
                    continue;
                }

                let mut next_cube = self.padded_cube.clone();
                next_cube.set(d1, d2, d3).unwrap();
                return Some(next_cube);
            } else {
                return None;
            }
        }
    }
}

impl PolyCube {
    pub fn expand(&self) -> impl Iterator<Item = PolyCube> + Clone {
        ToggleIterator {
            dim_1: 1,
            dim_2: 1,
            dim_3: 1,
            iterating_cube: None,
            padded_cube: self.pad_one(),
            done: false,
        }
    }
}

#[test]
pub fn correct_amount_of_rotations() {
    let cube = PolyCube::new_equal_sides(5);

    assert_eq!(cube.clone().rotations_in_plane((1, 2)).count(), 4);
    assert_eq!(cube.all_rotations().count(), 24);
}
