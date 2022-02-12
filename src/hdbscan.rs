use ndarray::{Array1, ArrayBase, ArrayView1, ArrayView2, Data, Ix2};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::convert::TryFrom;
use std::fmt::Debug;
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign, Sub};
use succinct::{BitVecMut, BitVector};

use super::Fit;
use petal_neighbors::distance::{Euclidean, Metric};
use petal_neighbors::BallTree;

#[derive(Debug, Deserialize, Serialize)]
pub struct HDbscan<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,
    pub alpha: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub metric: M,
}

impl<A> Default for HDbscan<A, Euclidean>
where
    A: Float,
{
    #[must_use]
    fn default() -> Self {
        Self {
            eps: A::from(0.5_f32).expect("valid float"),
            alpha: A::one(),
            min_samples: 15,
            min_cluster_size: 15,
            metric: Euclidean::default(),
        }
    }
}

impl<S, A, M> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for HDbscan<A, M>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync + TryFrom<u32>,
    <A as std::convert::TryFrom<u32>>::Error: Debug,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync,
{
    fn fit(&mut self, input: &ArrayBase<S, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        if input.is_empty() {
            return (HashMap::new(), Vec::new());
        }
        let db = BallTree::new(input.view(), self.metric.clone()).expect("non-empty array");
        let core_distances = Array1::from_vec(
            input
                .rows()
                .into_iter()
                .map(|r| {
                    db.query(&r, self.min_samples)
                        .1
                        .last()
                        .copied()
                        .expect("at least one point should be returned")
                })
                .collect(),
        );
        let mut mst = mst_linkage(
            input.view(),
            &self.metric,
            core_distances.view(),
            self.alpha,
        )
        .into_raw_vec();
        mst.sort_unstable_by(|a, b| a.2.partial_cmp(&(b.2)).expect("invalid distance"));
        let sorted_mst = Array1::from_vec(mst);
        let labeled = label(sorted_mst);
        let condensed = Array1::from_vec(condense_mst(labeled.view(), self.min_cluster_size));
        find_clusters(&condensed.view())
    }
}

fn mst_linkage<A: Float>(
    input: ArrayView2<A>,
    metric: &dyn Metric<A>,
    core_distances: ArrayView1<A>,
    alpha: A,
) -> Array1<(usize, usize, A)> {
    let nrows = input.nrows();

    assert_eq!(
        nrows,
        core_distances.len(),
        "dimensions of distance_metric and core_distances should match"
    );

    assert!(
        nrows >= 2,
        "dimensions of distance_metric and core_distances should be greater than 1"
    );

    let mut mst = Array1::<(usize, usize, A)>::uninit(nrows - 1);
    let mut in_tree: Vec<bool> = vec![false; nrows];
    let mut cur = 0;
    // edge uv: shortest_edges[v] = (mreachability_as_||uv||, u)
    // shortest as in shortest edges to v among  all nodes currently in tree
    let mut shortest_edges: Vec<(A, usize)> = vec![(A::max_value(), 1); nrows];

    for i in 0..nrows - 1 {
        // Add `cur` to tree
        in_tree[cur] = true;
        let core_cur = core_distances[cur];

        // next edge to add to tree
        let mut source: usize = 0;
        let mut next: usize = 0;
        let mut distance = A::max_value();

        for j in 0..nrows {
            if in_tree[j] {
                // skip if j is already in the tree
                continue;
            }

            let right = shortest_edges[j];
            let mut left = (metric.distance(&input.row(cur), &input.row(j)), cur);

            if alpha != A::from(1).expect("conversion failure") {
                left.0 = left.0 / alpha;
            } // convert distance matrix to `distance_metric / alpha` ?

            let core_j = core_distances[j];

            // right < MReachability_cur_j
            if (right.0 < core_cur || right.0 < left.0 || right.0 < core_j) && right.0 < distance {
                next = j;
                distance = right.0;
                source = right.1;
            }

            let tmp = if core_j > core_cur { core_j } else { core_cur };
            if tmp > left.0 {
                left.0 = tmp;
            }

            if left.0 < right.0 {
                shortest_edges[j] = left;
                if left.0 < distance {
                    distance = left.0;
                    source = left.1;
                    next = j;
                }
            } else if right.0 < distance {
                distance = right.0;
                source = right.1;
                next = j;
            }
        }

        mst[i] = MaybeUninit::new((source, next, distance)); // check MaybeUninit usage!
        cur = next;
    }

    unsafe { mst.assume_init() }
}

fn label<A: Float>(mst: Array1<(usize, usize, A)>) -> Array1<(usize, usize, A, usize)> {
    let n = mst.len() + 1;
    let mut uf = UnionFind::new(n);
    mst.into_iter()
        .map(|(mut a, mut b, delta)| {
            a = uf.fast_find(a);
            b = uf.fast_find(b);
            (a, b, delta, uf.union(a, b))
        })
        .collect()
}

fn condense_mst<A: Float + Div>(
    mst: ArrayView1<(usize, usize, A, usize)>,
    min_cluster_size: usize,
) -> Vec<(usize, usize, A, usize)> {
    let root = mst.len() * 2;
    let n = mst.len() + 1;

    let mut relabel = Array1::<usize>::uninit(root + 1);
    relabel[root] = MaybeUninit::new(n);
    let mut next_label = n + 1;
    let mut ignore = vec![false; root + 1];
    let mut result = Vec::new();

    let bsf = bfs_mst(mst, root);
    for node in bsf {
        if node < n {
            continue;
        }
        if ignore[node] {
            continue;
        }
        let info = mst[node - n];
        let lambda = if info.2 > A::zero() {
            A::one() / info.2
        } else {
            A::max_value()
        };
        let left = info.0;
        let left_count = if left < n { 1 } else { mst[left - n].3 };

        let right = info.1;
        let right_count = if right < n { 1 } else { mst[right - n].3 };

        match (
            left_count >= min_cluster_size,
            right_count >= min_cluster_size,
        ) {
            (true, true) => {
                relabel[left] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    left_count,
                ));
                next_label += 1;

                relabel[right] = MaybeUninit::new(next_label);
                result.push((
                    unsafe { relabel[node].assume_init() },
                    next_label,
                    lambda,
                    right_count,
                ));
                next_label += 1;
            }
            (true, false) => {
                relabel[left] = relabel[node];
                for child in bfs_mst(mst, right) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, true) => {
                relabel[right] = relabel[node];
                for child in bfs_mst(mst, left) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
            (false, false) => {
                for child in bfs_mst(mst, node).into_iter().skip(1) {
                    if child < n {
                        result.push((unsafe { relabel[node].assume_init() }, child, lambda, 1));
                    }
                    ignore[child] = true;
                }
            }
        }
    }
    result
}

fn get_stability<A: Float + AddAssign + Sub + TryFrom<u32>>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
) -> HashMap<usize, A>
where
    <A as TryFrom<u32>>::Error: Debug,
{
    let mut births: HashMap<_, _> = condensed_tree.iter().fold(HashMap::new(), |mut births, v| {
        let entry = births.entry(v.1).or_insert(v.2);
        if *entry > v.2 {
            *entry = v.2;
        }
        births
    });

    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("couldn't find the smallest cluster")
        .0;

    let entry = births.entry(min_parent).or_insert_with(A::zero);
    *entry = A::zero();

    condensed_tree.iter().fold(
        HashMap::new(),
        |mut stability, (parent, _child, lambda, size)| {
            let entry = stability.entry(*parent).or_insert_with(A::zero);
            let birth = births.get(parent).expect("invalid child node.");
            *entry += (*lambda - *birth)
                * A::try_from(u32::try_from(*size).expect("out of bound")).expect("out of bound");
            stability
        },
    )
}

fn find_clusters<A: Float + AddAssign + Sub + TryFrom<u32>>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>)
where
    <A as TryFrom<u32>>::Error: Debug,
{
    let mut stability = get_stability(condensed_tree);
    let mut nodes: Vec<_> = stability.keys().copied().collect();
    nodes.sort_unstable();
    nodes.reverse();
    nodes.remove(nodes.len() - 1);

    let tree: Vec<_> = condensed_tree
        .iter()
        .filter_map(|(p, c, _, s)| if *s > 1 { Some((*p, *c)) } else { None })
        .collect();

    let mut clusters: HashSet<_> = stability.keys().copied().collect();
    for node in nodes {
        let subtree_stability = tree.iter().fold(A::zero(), |acc, (p, c)| {
            if *p == node {
                acc + *stability.get(c).expect("corruptted stability dictionary")
            } else {
                acc
            }
        });

        stability.entry(node).and_modify(|v| {
            if *v < subtree_stability {
                clusters.remove(&node);
                *v = subtree_stability;
            } else {
                let bfs = bfs_tree(&tree, node);
                for child in bfs {
                    if child != node {
                        clusters.remove(&child);
                    }
                }
            }
        });
    }

    let mut clusters: Vec<_> = clusters.into_iter().collect();
    clusters.sort_unstable();
    let clusters: HashMap<_, _> = clusters
        .into_iter()
        .enumerate()
        .map(|(id, c)| (c, id))
        .collect();
    let max_parent = condensed_tree
        .iter()
        .max_by_key(|v| v.0)
        .expect("no maximum parent available")
        .0;
    let min_parent = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("no minimum parent available")
        .0;

    let mut uf = TreeUnionFind::new(max_parent + 1);
    for (parent, child, _, _) in condensed_tree {
        if !clusters.contains_key(child) {
            uf.union(*parent, *child);
        }
    }

    let mut res_clusters: HashMap<_, Vec<_>> = HashMap::new();
    let mut outliers = vec![];
    for n in 0..min_parent {
        let cluster = uf.find(n);
        if cluster > min_parent {
            let c = res_clusters.entry(cluster).or_default();
            c.push(n);
        } else {
            outliers.push(n);
        }
    }
    (res_clusters, outliers)
}

fn bfs_tree(tree: &[(usize, usize)], root: usize) -> Vec<usize> {
    let mut result = vec![];
    let mut to_process = HashSet::new();
    to_process.insert(root);
    while !to_process.is_empty() {
        result.extend(to_process.iter());
        to_process = tree
            .iter()
            .filter_map(|(p, c)| {
                if to_process.contains(p) {
                    Some(*c)
                } else {
                    None
                }
            })
            .collect::<HashSet<_>>();
    }
    result
}

fn bfs_mst<A: Float>(mst: ArrayView1<(usize, usize, A, usize)>, start: usize) -> Vec<usize> {
    let n = mst.len() + 1;

    let mut to_process = vec![start];
    let mut result = vec![];

    while !to_process.is_empty() {
        result.extend_from_slice(to_process.as_slice());
        to_process = to_process
            .into_iter()
            .filter_map(|x| {
                if x >= n {
                    Some(vec![mst[x - n].0, mst[x - n].1].into_iter())
                } else {
                    None
                }
            })
            .flatten()
            .collect();
    }
    result
}

#[allow(dead_code)]
#[derive(Debug)]
struct TreeUnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    is_component: BitVector<u64>,
}

#[allow(dead_code)]
impl TreeUnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..n).into_iter().collect();
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

    fn find(&mut self, x: usize) -> usize {
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

    fn union(&mut self, x: usize, y: usize) {
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

    fn components(&self) -> Vec<usize> {
        self.is_component
            .iter()
            .enumerate()
            .filter_map(|(idx, v)| if v { Some(idx) } else { None })
            .collect()
    }
}

struct UnionFind {
    parent: Vec<usize>,
    size: Vec<usize>,
    next_label: usize,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        let parent = (0..2 * n).into_iter().collect();
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

    fn union(&mut self, m: usize, n: usize) -> usize {
        self.parent[m] = self.next_label;
        self.parent[n] = self.next_label;
        let res = self.size[m] + self.size[n];
        self.size[self.next_label] = res;
        self.next_label += 1;
        res
    }

    fn fast_find(&mut self, mut n: usize) -> usize {
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
    fn hdbscan() {
        use crate::Fit;
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            eps: 0.5,
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
        };
        let (clusters, outliers) = hdbscan.fit(&data);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn mst_linkage() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::distance::Euclidean;
        //  0, 1, 2, 3, 4, 5, 6
        // {A, B, C, D, E, F, G}
        // {AB = 7, AD = 5,
        //  BC = 8, BD = 9, BE = 7,
        //  CB = 8, CE = 5,
        //  DB = 9, DE = 15, DF = 6,
        //  EF = 8, EG = 9
        //  FG = 11}
        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);
        let core_distances = arr1(&[5., 7., 5., 5., 5., 6., 9.]);
        let mst = super::mst_linkage(
            input.view(),
            &Euclidean::default(),
            core_distances.view(),
            1.,
        );
        let answer = arr1(&[
            (0, 3, 5.),
            (0, 1, 7.),
            (1, 5, 7.),
            (1, 2, 8.),
            (2, 4, 5.),
            (4, 6, 9.),
        ]);
        assert_eq!(mst, answer);
    }

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
        assert_eq!((0..3).into_iter().collect::<Vec<_>>(), uf.parent);
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

    #[test]
    fn label() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5.),
            (4, 2, 5.),
            (3, 5, 6.),
            (0, 1, 7.),
            (1, 4, 7.),
            (4, 6, 9.),
        ]);
        let labeled_mst = super::label(mst);
        assert_eq!(
            labeled_mst,
            arr1(&[
                (0, 3, 5., 2),
                (4, 2, 5., 2),
                (7, 5, 6., 3),
                (9, 1, 7., 4),
                (10, 8, 7., 6),
                (11, 6, 9., 7)
            ])
        );
    }

    #[test]
    fn bfs_mst() {
        use ndarray::arr1;
        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);
        let root = mst.len() * 2;
        let bfs = super::bfs_mst(mst.view(), root);
        assert_eq!(bfs, [12, 11, 6, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(mst.view(), 11);
        assert_eq!(bfs, vec![11, 10, 8, 9, 1, 4, 2, 7, 5, 0, 3]);

        let bfs = super::bfs_mst(mst.view(), 8);
        assert_eq!(bfs, vec![8, 4, 2]);
    }

    #[test]
    fn condense_mst() {
        use ndarray::arr1;

        let mst = arr1(&[
            (0, 3, 5., 2),
            (4, 2, 5., 2),
            (7, 5, 6., 3),
            (9, 1, 7., 4),
            (10, 8, 7., 6),
            (11, 6, 9., 7),
        ]);

        let condensed_mst = super::condense_mst(mst.view(), 3);
        assert_eq!(
            condensed_mst,
            vec![
                (7, 6, 1. / 9., 1),
                (7, 4, 1. / 7., 1),
                (7, 2, 1. / 7., 1),
                (7, 1, 1. / 7., 1),
                (7, 5, 1. / 6., 1),
                (7, 0, 1. / 6., 1),
                (7, 3, 1. / 6., 1)
            ],
        );
    }

    #[test]
    fn get_stability() {
        use ndarray::arr1;
        use std::collections::HashMap;

        let condensed = arr1(&[
            (7, 6, 1. / 9., 1),
            (7, 4, 1. / 7., 1),
            (7, 2, 1. / 7., 1),
            (7, 1, 1. / 7., 1),
            (7, 5, 1. / 6., 1),
            (7, 0, 1. / 6., 1),
            (7, 3, 1. / 6., 1),
        ]);
        let stability_map = super::get_stability(&condensed.view());
        let mut answer = HashMap::new();
        answer.insert(7, 1. / 9. + 3. / 7. + 3. / 6.);
        assert_eq!(stability_map, answer);
    }
}
