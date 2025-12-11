use std::convert::TryFrom;
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign};

use itertools::Itertools;
use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::distance::Metric;
use petal_neighbors::BallTree;
use rayon::prelude::*;

use crate::union_find::TreeUnionFind;

#[allow(clippy::needless_pass_by_value)] // Silences clippy warning. TODO: Update the parameter type to [`ArrayRef`](https://docs.rs/ndarray/latest/ndarray/struct.ArrayRef.html).
pub fn mst_linkage<A: FloatCore>(
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

    if nrows == 0 {
        // If there are no input points, return an empty MST.
        return Array1::from_vec(vec![]);
    }

    if nrows == 1 {
        // If there is only one input point, return a single edge with zero distance.
        return Array1::from_vec(vec![(0, 0, A::zero())]);
    }

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

pub fn condense_mst<A: FloatCore + Div>(
    mst: &[(usize, usize, A, usize)],
    min_cluster_size: usize,
) -> Vec<(usize, usize, A, usize)> {
    // min_parent gives the number of points in the hierarchy
    let n = mst
        .iter()
        .map(|(parent, _, _, _)| *parent)
        .min()
        .map_or(0, |min_parent| min_parent);

    // max_parent gives the number of clusters in the hierarchy
    let max_parent = mst
        .iter()
        .map(|(parent, _, _, _)| *parent)
        .max()
        .map_or(0, |max_parent| max_parent);

    let mut result: Vec<(usize, usize, A, usize)> = Vec::new();

    // Start with every node having the root label
    let mut label = vec![n; max_parent + 1];

    // Keep the minimum density level of cluster formations w.r.t. the minimum cluster size
    let mut lambda = vec![A::max_value(); max_parent + 1];

    // Top down pass to relabel the nodes w.r.t. the minimum cluster size
    let mut next_label: usize = n + 1;
    for ((parent, eps), edges) in &mst
        .iter()
        .rev()
        .chunk_by(|(parent, _, eps, _)| (*parent, *eps))
    {
        let edges = edges
            .map(|(_, child, _, child_size)| (*child, *child_size))
            .collect::<Vec<_>>();

        // Update the lambda value for the parent cluster
        let parent_size = edges
            .iter()
            .map(|(_, child_size)| *child_size)
            .sum::<usize>();
        if parent_size >= min_cluster_size {
            lambda[parent] = if eps > A::zero() {
                A::one() / eps
            } else {
                A::max_value()
            };
        }

        // Partition the children into new clusters and non-clusters
        let (mut new_clusters, mut non_clusters): (Vec<_>, Vec<_>) = edges
            .into_iter()
            .partition(|&(_, child_size)| child_size >= min_cluster_size);

        // If the parent is not splitting into 2 or more new clusters,
        // then parent is shrinking, so no need to create a new label
        if new_clusters.len() <= 1 {
            non_clusters.extend_from_slice(&new_clusters);
            new_clusters.clear();
        }

        // Assigning new labels to the child clusters
        for (child, child_size) in new_clusters {
            label[child] = next_label;
            result.push((label[parent], next_label, lambda[parent], child_size));
            next_label += 1;
        }

        // Propogate the parent's label and lambda to the non-cluster children:
        for (child, child_size) in non_clusters {
            label[child] = label[parent];
            lambda[child] = lambda[parent];
            if child_size == 1 {
                // The child is a single point, add it to the result
                result.push((label[parent], child, lambda[parent], child_size));
            }
        }
    }
    result
}

#[allow(dead_code)]
pub struct Boruvka<'a, A, M>
where
    A: FloatCore,
    M: Metric<A>,
{
    db: BallTree<'a, A, M>,
    min_samples: usize,
    candidates: Candidates<A>,
    components: Components,
    core_distances: Array1<A>,
    bounds: Vec<A>,
    mst: Vec<(usize, usize, A)>,
}

#[allow(dead_code)]
impl<'a, A, M> Boruvka<'a, A, M>
where
    A: FloatCore + AddAssign + DivAssign + FromPrimitive + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    pub fn new(db: BallTree<'a, A, M>, min_samples: usize) -> Self {
        let mut candidates = Candidates::new(db.points.nrows());
        let components = Components::new(db.nodes.len(), db.points.nrows());
        let bounds = vec![A::max_value(); db.nodes.len()];
        let core_distances = compute_core_distances(&db, min_samples, &mut candidates);
        let mst = Vec::with_capacity(db.points.nrows() - 1);
        Boruvka {
            db,
            min_samples,
            candidates,
            components,
            core_distances,
            bounds,
            mst,
        }
    }

    pub fn min_spanning_tree(mut self) -> Array1<(usize, usize, A)> {
        let mut num_components = self.update_components();

        while num_components > 1 {
            self.traversal(0, 0);
            num_components = self.update_components();
        }
        Array1::from_vec(self.mst)
    }

    fn update_components(&mut self) -> usize {
        let components = self.components.get_current();
        for i in components {
            let Some((src, sink, dist)) = self.candidates.get(i) else {
                continue;
            };

            if self.components.add(src, sink).is_none() {
                self.candidates.reset(i);
                continue;
            }

            self.candidates.distances[i] = A::max_value();

            self.mst.push((src, sink, dist));

            if self.mst.len() == self.db.num_points() - 1 {
                return self.components.len();
            }
        }
        self.components.update_points();
        for n in (0..self.db.num_nodes()).rev() {
            match self.db.children_of(n) {
                None => {
                    let mut points = self
                        .db
                        .points_of(n)
                        .iter()
                        .map(|i| self.components.point[*i]);
                    let pivot = points.next().expect("empty node");
                    if points.all(|c| c == pivot) {
                        self.components.node[n] =
                            u32::try_from(pivot).expect("overflow components");
                    }
                }
                Some((left, right)) => {
                    if self.components.node[left] == self.components.node[right]
                        && self.components.node[left] != u32::MAX
                    {
                        self.components.node[n] = self.components.node[left];
                    }
                }
            }
        }
        self.reset_bounds();
        self.components.len()
    }

    fn traversal(&mut self, query: usize, reference: usize) {
        // prune min{||query - ref||} >= bound_query
        let node_dist = self.db.node_distance_lower_bound(query, reference);
        if node_dist >= self.bounds[query] {
            return;
        }
        // prune when query and ref are in the same component
        if self.components.node[query] == self.components.node[reference]
            && self.components.node[query] != u32::MAX
        {
            return;
        }

        let query_children = self.db.children_of(query);
        let ref_children = self.db.children_of(reference);
        match (
            query_children,
            ref_children,
            self.db.compare_nodes(query, reference),
        ) {
            (None, None, _) => {
                let mut upper = A::zero();
                for &i in self.db.points_of(query) {
                    let c1 = self.components.point[i];
                    // mreach(i, j) >= core_i > candidate[c1]
                    // i.e. current best candidate for component c1 => prune
                    if self.core_distances[i] > self.candidates.distances[c1] {
                        continue;
                    }
                    for &j in self.db.points_of(reference) {
                        let c2 = self.components.point[j];
                        // mreach(i, j) >= core_j > candidate[c1] => prune
                        // i, j in the same component => prune
                        if self.core_distances[j] > self.candidates.distances[c1] || c1 == c2 {
                            continue;
                        }

                        let mut mreach = self
                            .db
                            .metric
                            .distance(&self.db.points.row(i), &self.db.points.row(j));
                        if self.core_distances[j] > mreach {
                            mreach = self.core_distances[j];
                        }
                        if self.core_distances[i] > mreach {
                            mreach = self.core_distances[i];
                        }

                        if mreach < self.candidates.distances[c1] {
                            self.candidates.update(c1, (i, j, mreach));
                        }
                    }
                    if self.candidates.distances[c1] > upper {
                        upper = self.candidates.distances[c1];
                    }
                }

                // Use only the upper bound (max candidate distance) for
                // simplicity and performance.
                if upper < self.bounds[query] {
                    self.bounds[query] = upper;
                    let mut cur = query;
                    while cur > 0 {
                        let p = (cur - 1) / 2;
                        let new_bound = self.bound(p);
                        if new_bound >= self.bounds[p] {
                            break;
                        }
                        self.bounds[p] = new_bound;
                        cur = p;
                    }
                }
            }
            (None, Some((left, right)), _)
            | (_, Some((left, right)), Some(std::cmp::Ordering::Less)) => {
                let left_bound = self.db.node_distance_lower_bound(query, left);
                let right_bound = self.db.node_distance_lower_bound(query, right);

                if left_bound < right_bound {
                    self.traversal(query, left);
                    self.traversal(query, right);
                } else {
                    self.traversal(query, right);
                    self.traversal(query, left);
                }
            }
            (Some((left, right)), _, _) => {
                let left_bound = self.db.node_distance_lower_bound(reference, left);
                let right_bound = self.db.node_distance_lower_bound(reference, right);
                if left_bound < right_bound {
                    self.traversal(left, reference);
                    self.traversal(right, reference);
                } else {
                    self.traversal(right, reference);
                    self.traversal(left, reference);
                }
            }
        }
    }

    fn reset_bounds(&mut self) {
        self.bounds.iter_mut().for_each(|v| *v = A::max_value());
    }

    #[inline]
    fn bound(&self, parent: usize) -> A {
        let left = 2 * parent + 1;
        let right = left + 1;

        // Use only upper bound (max of children)
        if self.bounds[left] > self.bounds[right] {
            self.bounds[left]
        } else {
            self.bounds[right]
        }
    }
}

// core_distances: distance of center to min_samples' closest point (including the center).
fn compute_core_distances<A, M>(
    db: &BallTree<A, M>,
    min_samples: usize,
    candidates: &mut Candidates<A>,
) -> Array1<A>
where
    A: AddAssign + DivAssign + FromPrimitive + FloatCore + Sync + Send,
    M: Metric<A> + Sync + Send,
{
    let mut knn_indices = vec![0; db.points.nrows() * min_samples];
    let mut core_distances = vec![A::zero(); db.points.nrows()];
    let rows: Vec<(usize, (&mut [usize], &mut A))> = knn_indices
        .chunks_mut(min_samples)
        .zip(core_distances.iter_mut())
        .enumerate()
        .collect();
    rows.into_par_iter().for_each(|(i, (indices, dist))| {
        let row = db.points.row(i);
        let (idx, d) = db.query(&row, min_samples);
        indices.clone_from_slice(&idx);
        *dist = *d.last().expect("ball tree query failed");
    });

    knn_indices
        .chunks_exact(min_samples)
        .enumerate()
        .for_each(|(n, row)| {
            for val in row.iter().skip(1).rev() {
                if core_distances[*val] <= core_distances[n] {
                    candidates.update(n, (n, *val, core_distances[n]));
                }
            }
        });

    Array1::from_vec(core_distances)
}

#[allow(dead_code)]
struct Candidates<A> {
    points: Vec<u32>,
    neighbors: Vec<u32>,
    distances: Vec<A>,
}

#[allow(dead_code)]
impl<A: FloatCore> Candidates<A> {
    fn new(n: usize) -> Self {
        // define max_value as NULL
        let neighbors = vec![u32::MAX; n];
        // define max_value as NULL
        let points = vec![u32::MAX; n];
        // define max_value as infinite far
        let distances = vec![A::max_value(); n];
        Self {
            points,
            neighbors,
            distances,
        }
    }

    fn get(&self, i: usize) -> Option<(usize, usize, A)> {
        if self.is_undefined(i) {
            None
        } else {
            Some((
                usize::try_from(self.points[i]).expect("fail to convert points"),
                usize::try_from(self.neighbors[i]).expect("fail to convert neighbor"),
                self.distances[i],
            ))
        }
    }

    fn update(&mut self, i: usize, val: (usize, usize, A)) {
        self.distances[i] = val.2;
        self.points[i] = u32::try_from(val.0).expect("candidate index overflow");
        self.neighbors[i] = u32::try_from(val.1).expect("candidate index overflow");
    }

    fn reset(&mut self, i: usize) {
        self.points[i] = u32::MAX;
        self.neighbors[i] = u32::MAX;
        self.distances[i] = A::max_value();
    }

    fn is_undefined(&self, i: usize) -> bool {
        self.points[i] == u32::MAX || self.neighbors[i] == u32::MAX
    }
}

#[allow(dead_code)]
struct Components {
    point: Vec<usize>,
    node: Vec<u32>,
    uf: TreeUnionFind,
}

#[allow(dead_code)]
impl Components {
    fn new(m: usize, n: usize) -> Self {
        // each point started as its own component.
        let point = (0..n).collect();
        // the component of the node is concluded when
        // all the enclosed points are in the same component
        let node = vec![u32::MAX; m];
        let uf = TreeUnionFind::new(n);
        Self { point, node, uf }
    }

    fn add(&mut self, src: usize, sink: usize) -> Option<()> {
        let current_src = self.uf.find(src);
        let current_sink = self.uf.find(sink);
        if current_src == current_sink {
            return None;
        }
        self.uf.union(current_src, current_sink);
        Some(())
    }

    fn update_points(&mut self) {
        for i in 0..self.point.len() {
            self.point[i] = self.uf.find(i);
        }
    }

    fn get_current(&self) -> Vec<usize> {
        self.uf.components()
    }

    fn len(&self) -> usize {
        self.uf.num_components()
    }
}

mod test {

    #[test]
    fn condense_mst() {
        // Given the following hierarchy of 7 points:
        //             12
        //           /    \        <-- eps = 8.0
        //         10       11
        //        /  \      / \    <-- eps = 4.0
        //       7    8    9   6
        //      /|    |\   |\      <-- eps = 2.0
        //     0 1    2 3  4 5

        let mst = vec![
            (7, 0, 2., 1),
            (7, 1, 2., 1),
            (8, 2, 2., 1),
            (8, 3, 2., 1),
            (9, 4, 2., 1),
            (9, 5, 2., 1),
            (10, 7, 4., 2),
            (10, 8, 4., 2),
            (11, 9, 4., 2),
            (11, 6, 4., 1),
            (12, 10, 8., 4),
            (12, 11, 8., 3),
        ];
        let min_cluster_size = 3;

        // Condensing the hierarchy based on the minimum cluster size = 3 should yield:
        //             7
        //           /   \
        //         9       8
        //       // \\    /|\
        //      0 1 3 4  4 5 6

        let condensed = super::condense_mst(&mst, min_cluster_size);
        assert_eq!(
            condensed,
            vec![
                (7, 8, 1. / 8., 3),
                (7, 9, 1. / 8., 4),
                (8, 6, 1. / 4., 1),
                (8, 5, 1. / 4., 1),
                (8, 4, 1. / 4., 1),
                (9, 3, 1. / 4., 1),
                (9, 2, 1. / 4., 1),
                (9, 1, 1. / 4., 1),
                (9, 0, 1. / 4., 1),
            ]
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
    fn boruvka() {
        use ndarray::{arr1, arr2};
        use petal_neighbors::{distance::Euclidean, BallTree};

        let input = arr2(&[
            [0., 0.],
            [7., 0.],
            [15., 0.],
            [0., -5.],
            [15., -5.],
            [7., -7.],
            [15., -14.],
        ]);

        let db = BallTree::new(input, Euclidean::default()).unwrap();
        let boruvka = super::Boruvka::new(db, 2);
        let mst = boruvka.min_spanning_tree();

        let answer = arr1(&[
            (0, 3, 5.0),
            (1, 0, 7.0),
            (2, 4, 5.0),
            (5, 1, 7.0),
            (6, 4, 9.0),
            (1, 2, 8.0),
        ]);
        assert_eq!(answer, mst);
    }

    /// Verifies that Boruvka and Prim's algorithms compute consistent MSTs.
    #[test]
    fn boruvka_prim_mst_consistency() {
        use ndarray::Array2;
        use petal_neighbors::distance::Euclidean;
        use petal_neighbors::BallTree;

        // Generate a dataset to trigger the bug in issue #69. The bug requires
        // specific tree structure conditions where all points in a leaf node
        // get skipped.
        let n_points = 4530;
        let n_dims = 12;
        let min_samples = 15;

        // Generate deterministic pseudo-random data using a simple LCG
        let mut seed: u64 = 42;
        let mut data_vec = Vec::with_capacity(n_points * n_dims);
        for _ in 0..(n_points * n_dims) {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            #[allow(clippy::cast_precision_loss)]
            let val = (seed >> 33) as f64 / (1u64 << 31) as f64 * 20.0 - 10.0;
            data_vec.push(val);
        }

        let data: Array2<f64> =
            Array2::from_shape_vec((n_points, n_dims), data_vec).expect("valid shape");

        let metric = Euclidean::default();

        // Compute MST using Boruvka
        let db_boruvka = BallTree::new(data.view(), metric.clone()).expect("non-empty array");
        let boruvka = super::Boruvka::new(db_boruvka, min_samples);
        let mst_boruvka = boruvka.min_spanning_tree();
        let weight_boruvka: f64 = mst_boruvka.iter().map(|(_, _, w)| *w).sum();

        // Compute MST using Prim (mst_linkage)
        let db_prim = BallTree::new(data.view(), metric.clone()).expect("non-empty array");
        let core_distances = ndarray::Array1::from_vec(
            data.rows()
                .into_iter()
                .map(|r| {
                    db_prim
                        .query(&r, min_samples)
                        .1
                        .last()
                        .copied()
                        .expect("at least one point")
                })
                .collect(),
        );
        let mst_prim = super::mst_linkage(data.view(), &metric, core_distances.view(), 1.0);
        let weight_prim: f64 = mst_prim.iter().map(|(_, _, w)| *w).sum();

        // Both algorithms should produce MSTs with the same total weight.
        // The MST is unique when all edge weights are distinct, but even with
        // ties, the total weight should be identical.
        assert!(
            (weight_boruvka - weight_prim).abs() < 1e-10,
            "MST weights differ: Boruvka={weight_boruvka}, Prim={weight_prim}"
        );
    }
}
