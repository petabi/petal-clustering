use std::cmp::Ordering;
use std::convert::TryFrom;
use std::fmt::Debug;

use succinct::{BitVecMut, BitVector};

#[allow(dead_code)]
#[derive(Debug)]
pub struct TreeUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    is_component: BitVector<u64>,
}

#[allow(dead_code)]
impl TreeUnionFind {
    pub fn new(n: usize) -> Self {
        let parent = (0..n).collect();
        let size = vec![0; n];
        let is_component = BitVector::with_fill(
            u64::try_from(n).expect("fail to build a large enough bit vector"),
            true,
        );
        Self {
            parent,
            size,
            is_component,
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        assert!(x < self.parent.len());
        if x != self.parent[x] {
            self.parent[x] = self.find(self.parent[x]);
            self.is_component.set_bit(
                u64::try_from(x).expect("fail to convert usize to u64"),
                false,
            );
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let xx = self.find(x);
        let yy = self.find(y);

        match self.size[xx].cmp(&self.size[yy]) {
            Ordering::Greater => self.parent[yy] = xx,
            Ordering::Equal => {
                self.parent[yy] = xx;
                self.size[xx] += 1;
            }
            Ordering::Less => self.parent[xx] = yy,
        }
    }

    pub fn components(&self) -> Vec<usize> {
        self.is_component
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| if v { Some(idx) } else { None })
            .collect()
    }

    pub fn num_components(&self) -> usize {
        self.is_component.iter().filter(|b| *b).count()
    }
}

mod test {

    #[test]
    fn tree_union_find() {
        use succinct::{BitVecMut, BitVector};

        let parent = vec![0, 0, 1, 2, 4];
        let size = vec![0; 5];
        let is_component = BitVector::with_fill(5, true);
        let mut uf = super::TreeUnionFind {
            parent,
            size,
            is_component,
        };
        assert_eq!(0, uf.find(3));
        assert_eq!(vec![0, 0, 0, 0, 4], uf.parent);
        uf.union(4, 0);
        assert_eq!(vec![4, 0, 0, 0, 4], uf.parent);
        assert_eq!(vec![0, 0, 0, 0, 1], uf.size);
        let mut bv = BitVector::with_fill(5, false);
        bv.set_bit(0, true);
        bv.set_bit(4, true);
        assert_eq!(bv, uf.is_component);
        assert_eq!(vec![0, 4], uf.components());

        uf = super::TreeUnionFind::new(3);
        assert_eq!((0..3).collect::<Vec<_>>(), uf.parent);
        assert_eq!(vec![0; 3], uf.size);
    }
}
