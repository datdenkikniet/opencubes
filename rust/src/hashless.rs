use crossbeam_channel::Sender;
use hashbrown::HashSet;

use crate::polycubes::{
    pcube::RawPCube,
    point_list::{CubeMapPos, Dim},
};

pub struct MapStore<const N: usize> {
    inner: HashSet<CubeMapPos<N>>,
}

impl<const N: usize> MapStore<N> {
    pub fn new() -> Self {
        Self {
            inner: HashSet::new(),
        }
    }

    /// helper function to not duplicate code for canonicalising polycubes
    /// and storing them in the hashset
    fn insert_map(&mut self, dim: Dim, map: CubeMapPos<N>, count: usize) {
        if !self.inner.contains(&map) {
            let map = map.to_min_rot_points(dim, count);
            self.inner.insert(map);
        }
    }

    /// Calculate the amount of canonical children of size `target`
    /// that polycube `seed` of size `count` has.
    ///
    /// This function does not store variants of the polycubes that
    /// it enumerates, it just keeps the count. This way, memory
    /// overhead is minimal.
    // TODO: improve this name once we unify this and pointslist
    pub fn enumerate_canonical_children_min_mem(
        seed: &CubeMapPos<N>,
        count: usize,
        target: usize,
    ) -> usize {
        let mut store = Self::new();
        let shape = seed.extrapolate_dim();

        let seed = seed.to_min_rot_points(shape, count);
        let shape = seed.extrapolate_dim();

        seed.expand(shape, count)
            .for_each(|(dim, count, map)| store.insert_map(dim, map, count));

        store
            .inner
            .retain(|child| child.is_canonical_root(count, &seed));

        if count + 1 == target {
            store.inner.len()
        } else {
            store
                .inner
                .iter()
                .map(|child| Self::enumerate_canonical_children_min_mem(child, count + 1, target))
                .sum()
        }
    }

    pub fn enumerate_canonical_children_min_mem_with_output(
        seed: &CubeMapPos<N>,
        count: usize,
        target: usize,
        sender: &Sender<RawPCube>,
    ) -> usize {
        let mut store = Self::new();
        let shape = seed.extrapolate_dim();

        let seed = seed.to_min_rot_points(shape, count);
        let shape = seed.extrapolate_dim();

        seed.expand(shape, count)
            .for_each(|(dim, count, map)| store.insert_map(dim, map, count));

        store
            .inner
            .retain(|child| child.is_canonical_root(count, &seed));

        if count + 1 == target {
            store.inner.iter().for_each(|c| {
                sender.send(c.into()).unwrap();
            });

            store.inner.len()
        } else {
            store
                .inner
                .iter()
                .map(|child| {
                    Self::enumerate_canonical_children_min_mem_with_output(
                        child,
                        count + 1,
                        target,
                        sender,
                    )
                })
                .sum()
        }
    }
}
