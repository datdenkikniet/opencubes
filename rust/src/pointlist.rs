use crate::polycubes::point_list::{CubeMapPos, Dim};

use hashbrown::{HashMap, HashSet};
use parking_lot::RwLock;
use rayon::prelude::*;

/// Structure to store sets of polycubes
pub struct MapStore {
    /// Stores the shape and fist block index as a key
    /// to the set of 15 block tails that correspond to that shape and start.
    /// used for reducing rwlock pressure on insertion
    /// used as buckets for parallelising
    /// however both of these give suboptomal performance due to the uneven distribution
    inner: HashMap<(Dim, u16), RwLock<HashSet<CubeMapPos<15>>>>,
}

impl MapStore {
    pub fn new() -> Self {
        Self {
            inner: HashMap::new(),
        }
    }

    pub fn insert_key(&mut self, shape: Dim, start_cube: u16) {
        if !self.inner.contains_key(&(shape, start_cube)) {
            self.inner
                .insert((shape, start_cube), RwLock::new(HashSet::new()));
        }
    }

    pub fn insert(&self, dim: Dim, map: CubeMapPos<16>, count: usize) {
        // Check if we don't already happen to be in the minimum rotation position.
        let mut body_maybemin = CubeMapPos::new();
        body_maybemin.cubes[0..count].copy_from_slice(&map.cubes[1..count + 1]);
        let dim_maybe = map.extrapolate_dim();

        // Weirdly enough, doing the copy and doing the lookup check this
        // way is faster than only copying if `inner` has en entry for
        // dim_maybe.
        if self
            .inner
            .get(&(dim_maybe, map.cubes[0]))
            .map(|v| v.read().contains(&body_maybemin))
            == Some(true)
        {
            return;
        }

        let map = map.to_min_rot_points(dim, count);

        let mut body = CubeMapPos::new();
        body.cubes[0..count].copy_from_slice(&map.cubes[1..count + 1]);

        let entry = self
            .inner
            .get(&(dim, map.cubes[0]))
            .expect("Cube size does not have entry in destination map");

        entry.write().insert(body);
    }

    /// helper for inner_exp in expand_cube_set it didnt like going directly in the closure
    fn expand_cube_sub_set(
        &self,
        shape: Dim,
        first_cube: u16,
        body: impl Iterator<Item = CubeMapPos<15>>,
        count: usize,
    ) {
        let mut seed = CubeMapPos {
            cubes: [first_cube, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        };

        for seed_body in body {
            for i in 1..count {
                seed.cubes[i] = seed_body.cubes[i - 1];
            }

            // body.cubes.copy_within(0..body.cubes.len() - 1, 1);

            seed.expand(shape, count)
                .for_each(|(dim, count, map)| self.insert(dim, map, count));
        }
    }

    pub fn expand_cube_set(self, count: usize, parallel: bool) -> Self {
        let mut dst = MapStore::new();

        // set up the dst sets before starting parallel processing so accessing doesnt block a global mutex
        for x in 0..=count + 1 {
            for y in 0..=(count + 1) / 2 {
                for z in 0..=(count + 1) / 3 {
                    for i in 0..(y + 1) * 32 {
                        dst.inner
                            .insert((Dim { x, y, z }, i as u16), RwLock::new(HashSet::new()));
                    }
                }
            }
        }

        let inner_exp = |((shape, first_cube), body): (_, RwLock<HashSet<_>>)| {
            dst.expand_cube_sub_set(shape, first_cube, body.into_inner().into_iter(), count);
        };

        // Use parallel iterator or not to run expand_cube_set
        if parallel {
            self.inner.into_par_iter().for_each(inner_exp);
        } else {
            self.inner.into_iter().for_each(inner_exp);
        }

        //retain only subsets that have polycubes
        dst.inner.retain(|_, v| v.read().len() > 0);

        dst
    }

    /// Count the number of polycubes across all subsets
    pub fn count_polycubes(&self) -> usize {
        let mut total = 0;
        #[cfg(feature = "diagnostics")]
        for ((d, s), body) in maps.iter().rev() {
            println!(
                "({}, {}, {}) {} _> {}",
                d.x + 1,
                d.y + 1,
                d.z + 1,
                s,
                body.len()
            );
        }
        for (_, body) in self.inner.iter() {
            total += body.read().len()
        }

        total
    }

    /// Destructively move the data from hashset to vector
    pub fn into_map_iter(self) -> impl Iterator<Item = CubeMapPos<16>> {
        self.inner.into_iter().flat_map(|((_, head), body)| {
            body.into_inner().into_iter().map(move |v| {
                let mut pos = CubeMapPos::new();
                pos.cubes[0] = head;
                pos.cubes[1..16].copy_from_slice(&v.cubes[0..15]);
                pos
            })
        })
    }

    /// Copy the data from hashset to vector
    pub fn to_vec(&self) -> Vec<CubeMapPos<16>> {
        let mut v = Vec::with_capacity(self.count_polycubes());

        for ((_, head), body) in self.inner.iter() {
            let bod = body.read();
            let mut cmp = CubeMapPos::new();
            cmp.cubes[0] = *head;
            for b in bod.iter() {
                for i in 0..15 {
                    cmp.cubes[i + 1] = b.cubes[i];
                }
                v.push(cmp);
            }
        }

        v
    }
}
