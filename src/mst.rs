use std::convert::TryFrom;
use std::mem::MaybeUninit;
use std::ops::{AddAssign, Div, DivAssign};

use ndarray::{Array1, ArrayView1, ArrayView2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::distance::Metric;
use petal_neighbors::BallTree;
use rayon::prelude::*;

use crate::union_find::TreeUnionFind;

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

fn bfs_mst<A: FloatCore>(mst: ArrayView1<(usize, usize, A, usize)>, start: usize) -> Vec<usize> {
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

pub fn condense_mst<A: FloatCore + Div>(
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
                let mut lower = A::max_value();
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
                    if self.candidates.distances[c1] < lower {
                        lower = self.candidates.distances[c1];
                    }
                    if self.candidates.distances[c1] > upper {
                        upper = self.candidates.distances[c1];
                    }
                }

                let radius = self.db.radius_of(query);
                let mut bound = lower + radius + radius;
                if bound > upper {
                    bound = upper;
                }
                if bound < self.bounds[query] {
                    self.bounds[query] = bound;
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
    fn lower_bound(&self, node: usize, parent: usize) -> A {
        let diff = self.db.radius_of(parent) - self.db.radius_of(node);
        self.bounds[node] + diff + diff
    }

    #[inline]
    fn bound(&self, parent: usize) -> A {
        let left = 2 * parent + 1;
        let right = left + 1;

        let upper = if self.bounds[left] > self.bounds[right] {
            self.bounds[left]
        } else {
            self.bounds[right]
        };

        let lower_left = self.lower_bound(left, parent);
        let lower_right = self.lower_bound(right, parent);
        let lower = if lower_left > lower_right {
            lower_right
        } else {
            lower_left
        };

        if lower > A::zero() && lower < upper {
            lower
        } else {
            upper
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
}
