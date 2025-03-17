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

pub struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        let parent = (0..2 * n).collect();
        let size = vec![1]
            .into_iter()
            .cycle()
            .take(n)
            .chain(vec![0].into_iter().cycle().take(n - 1))
            .collect();
        Self {
            parent,
            size,
            next_label: n,
        }
    }

    pub fn union(&mut self, m: usize, n: usize) -> usize {
        self.parent[m] = self.next_label;
        self.parent[n] = self.next_label;
        let res = self.size[m] + self.size[n];
        self.size[self.next_label] = res;
        self.next_label += 1;
        res
    }

    pub fn fast_find(&mut self, mut n: usize) -> usize {
        let mut root = n;
        while self.parent[n] != n {
            n = self.parent[n];
        }
        while self.parent[root] != n {
            let tmp = self.parent[root];
            self.parent[root] = n;
            root = tmp;
        }
        n
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

    #[test]
    fn union_find() {
        let mut uf = super::UnionFind::new(7);
        let pairs = vec![(0, 3), (4, 2), (3, 5), (0, 1), (1, 4), (4, 6)];
        let uf_res: Vec<_> = pairs
            .into_iter()
            .map(|(l, r)| {
                let ll = uf.fast_find(l);
                let rr = uf.fast_find(r);
                (ll, rr, uf.union(ll, rr))
            })
            .collect();
        assert_eq!(
            uf_res,
            vec![
                (0, 3, 2),
                (4, 2, 2),
                (7, 5, 3),
                (9, 1, 4),
                (10, 8, 6),
                (11, 6, 7)
            ]
        )
    }
}
