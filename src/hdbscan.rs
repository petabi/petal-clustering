use std::collections::HashMap;
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign, Sub};

use itertools::Itertools;
use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::distance::{Euclidean, Metric};
use petal_neighbors::BallTree;
use serde::{Deserialize, Serialize};

use super::Fit;
use crate::mst::{condense_mst, mst_linkage, Boruvka};
use crate::union_find::TreeUnionFind;

/// HDBSCAN (hierarchical density-based spatial clustering of applications with noise)
/// clustering algorithm.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use petal_neighbors::distance::Euclidean;
/// use petal_clustering::{HDbscan, Fit};
///
/// let points = array![
///             [1.0, 2.0],
///             [1.1, 2.2],
///             [0.9, 1.9],
///             [1.0, 2.1],
///             [-2.0, 3.0],
///             [-2.2, 3.1],
///         ];
/// let mut hdbscan = HDbscan {
///    alpha: 1.,
///    min_samples: 2,
///    min_cluster_size: 2,
///    metric: Euclidean::default(),
///    boruvka: false,
/// };
/// let (clusters, outliers, _outlier_scores) = hdbscan.fit(&points, None);
/// assert_eq!(clusters.len(), 2);   // two clusters found
///
/// assert_eq!(
///     outliers.len(),
///     points.nrows() - clusters.values().fold(0, |acc, v| acc + v.len()));
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct HDbscan<A, M> {
    /// The radius of a neighborhood.
    pub alpha: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub min_cluster_size: usize,
    pub metric: M,
    pub boruvka: bool,
}

impl<A> Default for HDbscan<A, Euclidean>
where
    A: FloatCore,
{
    fn default() -> Self {
        Self {
            alpha: A::one(),
            min_samples: 15,
            min_cluster_size: 15,
            metric: Euclidean::default(),
            boruvka: true,
        }
    }
}

/// Fits the HDBSCAN clustering algorithm to the given input data.
///
/// # Parameters
/// - `input`: A 2D array representing the dataset to cluster. Each row corresponds to a data point.
/// - `partial_labels`: An optional parameter for prelabelled data.
///
/// # Returns
/// A tuple containing:
/// - `HashMap<usize, Vec<usize>>`: A mapping of cluster IDs to the indices of points in each cluster.
/// - `Vec<usize>`: A vector of indices representing the noise points that do not belong to any cluster.
/// - `Vec<A>`: A vector of outlier scores for each data point.
///
/// # Notes
/// - The outlier scores are computed using the GLOSH algorithm.
/// - If `partial_labels` is provided, the algorithm will perform semi-supervised clustering using BC (`BCubed`) algorithm,
///   otherwise, it will perform unsupervised clustering using Excess of Mass (`EoM`) algorithm.
///
/// # References
/// - Campello, Ricardo JGB, et al. "Hierarchical density estimates for data clustering, visualization, and outlier detection."
///   ACM Transactions on Knowledge Discovery from Data (TKDD) 10.1 (2015): 1-51.
/// - Castro Gertrudes, Jadson, et al. "A unified view of density-based methods for semi-supervised clustering and classification."
///   Data mining and knowledge discovery 33.6 (2019): 1894-1952.
impl<S, A, M>
    Fit<
        ArrayBase<S, Ix2>,
        HashMap<usize, Vec<usize>>,
        (HashMap<usize, Vec<usize>>, Vec<usize>, Vec<A>),
    > for HDbscan<A, M>
where
    A: AddAssign + DivAssign + FloatCore + FromPrimitive + Sync + Send,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync + Send,
{
    fn fit(
        &mut self,
        input: &ArrayBase<S, Ix2>,
        partial_labels: Option<&HashMap<usize, Vec<usize>>>,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>, Vec<A>) {
        if input.is_empty() {
            return (HashMap::new(), Vec::new(), Vec::new());
        }
        let input = input.as_standard_layout();
        let db = BallTree::new(input.view(), self.metric.clone()).expect("non-empty array");

        let (mut mst, _offset) = if self.boruvka {
            let boruvka = Boruvka::new(db, self.min_samples);
            boruvka.min_spanning_tree().into_raw_vec_and_offset()
        } else {
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
            mst_linkage(
                input.view(),
                &self.metric,
                core_distances.view(),
                self.alpha,
            )
            .into_raw_vec_and_offset()
        };

        mst.sort_unstable_by(|a, b| a.2.partial_cmp(&(b.2)).expect("invalid distance"));
        let labeled = label(&mst);
        let condensed = condense_mst(&labeled, self.min_cluster_size);
        let outlier_scores = glosh(&condensed, self.min_cluster_size);
        let (clusters, outliers) =
            find_clusters(&Array1::from_vec(condensed).view(), partial_labels);
        (clusters, outliers, outlier_scores)
    }
}

fn label<A: FloatCore>(mst: &[(usize, usize, A)]) -> Vec<(usize, usize, A, usize)> {
    let n = mst.len() + 1;
    let mut result: Vec<(usize, usize, A, usize)> = Vec::with_capacity(2 * n);
    let mut next_label = n;
    let mut label = (0..2 * n).collect::<Vec<_>>(); // labels of subtrees
    let mut sizes = [vec![1; n], vec![0; n]].concat(); // sizes of subtrees
    let mut uf = TreeUnionFind::new(n);

    // HDBSCAN merges subtrees in the order of eps (distance)
    // where ties in eps should be merged at the same time:
    for (eps, edges) in &mst.iter().chunk_by(|(_, _, eps)| *eps) {
        let edges = edges.collect::<Vec<_>>();

        // Collect unique subtree roots (children)
        let subtree_roots = edges
            .iter()
            .flat_map(|(u, v, _)| [uf.find(*u), uf.find(*v)])
            .unique()
            .collect::<Vec<_>>();

        // Merge the subtrees
        for (u, v, _) in edges {
            uf.union(*u, *v);
        }

        // Assign parent-child labels
        let mut level: HashMap<usize, usize> = HashMap::new();
        for child in subtree_roots {
            let parent = uf.find(child);
            let parent_label = level.entry(parent).or_insert_with(|| {
                next_label += 1;
                next_label - 1
            });
            let child_label = label[child];
            result.push((*parent_label, child_label, eps, sizes[child_label]));
            sizes[*parent_label] += sizes[child_label];
            label[child] = *parent_label;
        }
    }
    result
}

fn get_stability<A: FloatCore + FromPrimitive + AddAssign + Sub>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
) -> HashMap<usize, A> {
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
            let Some(size) = A::from_usize(*size) else {
                panic!("invalid size");
            };
            *entry += (*lambda - *birth) * size;
            stability
        },
    )
}

fn get_bcubed<A: FloatCore + FromPrimitive + AddAssign + Sub>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
    partial_labels: &HashMap<usize, Vec<usize>>,
) -> HashMap<usize, A> {
    let num_labelled = partial_labels.values().fold(0, |acc, v| acc + v.len());

    // min_parent gives the number of events in the hierarchy
    let num_events = condensed_tree
        .iter()
        .map(|(parent, _, _, _)| *parent)
        .min()
        .map_or(0, |min_parent| min_parent);

    // initialize the labels with the partial labels (if any)
    let mut labels: Vec<Option<usize>> = vec![None; num_events];
    for (label, points) in partial_labels {
        for point in points {
            labels[*point] = Some(*label);
        }
    }

    let num_clusters = condensed_tree
        .iter()
        .map(|(parent, child, _, _)| parent.max(child))
        .max()
        .expect("empty condensed_mst");

    // bottom-up traverse the hierarchy to keep track of the counts of the labelled points
    // (same with the reverse order iteration on the condensed_mst)
    let mut label_map: HashMap<usize, HashMap<usize, A>> = HashMap::new();
    let mut num_labels: Vec<A> = vec![A::zero(); num_clusters + 1];
    let mut bcubed: Vec<A> = vec![A::zero(); num_clusters + 1];
    for (parent, child, _, _) in condensed_tree.iter().rev() {
        if *child < num_events {
            // point is labelled
            if let Some(label) = labels[*child] {
                let entry = label_map.entry(*parent).or_default();
                let count = entry.entry(label).or_insert(A::zero());
                *count += A::one();
                num_labels[*parent] += A::one();
            }
        } else {
            // extend with child cluster count map
            let child_map = label_map.remove(child).unwrap_or_default(); // remove to save space
            let child_num_labelled = num_labels[*child];

            let parent_map = label_map.entry(*parent).or_default();
            for (label, count) in child_map {
                // compute bcubed of the child cluster
                let precision = count / child_num_labelled;
                let recall = count / A::from(partial_labels[&label].len()).expect("invalid count");
                let fmeasure =
                    A::from(2).expect("invalid count") * precision * recall / (precision + recall);
                bcubed[*child] += count * fmeasure / A::from(num_labelled).expect("invalid count");

                // update the parent cluster label count map
                let c = parent_map.entry(label).or_insert(A::zero());
                *c += count;
                num_labels[*parent] += count;
            }
        }
    }

    condensed_tree
        .iter()
        .fold(HashMap::new(), |mut scores, (parent, _child, _, _)| {
            scores.entry(*parent).or_insert_with(|| bcubed[*parent]);
            scores
        })
}

fn find_clusters<A: FloatCore + FromPrimitive + AddAssign + Sub>(
    condensed_tree: &ArrayView1<(usize, usize, A, usize)>,
    partial_labels: Option<&HashMap<usize, Vec<usize>>>,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
    let mut stability = get_stability(condensed_tree);
    let mut bcubed = if let Some(partial_labels) = partial_labels {
        get_bcubed(condensed_tree, partial_labels)
    } else {
        HashMap::new()
    };

    let mut nodes: Vec<_> = stability.keys().copied().collect();
    nodes.sort_unstable();
    nodes.remove(0); // remove the root node

    let adj: HashMap<usize, Vec<usize>> =
        condensed_tree
            .iter()
            .fold(HashMap::new(), |mut adj, (p, c, _, _)| {
                adj.entry(*p).or_default().push(*c);
                adj
            });

    let num_clusters = condensed_tree
        .iter()
        .max_by_key(|v| v.0)
        .expect("no maximum parent available")
        .0;

    // bottom-up traverse the nodes to select the most top-level clusters
    let mut clusters: Vec<Option<usize>> = vec![None; num_clusters + 1];
    for node in nodes.iter().rev() {
        let subtree_stability = adj.get(node).map_or(A::zero(), |children| {
            children.iter().fold(A::zero(), |acc, c| {
                acc + *stability.get(c).unwrap_or(&A::zero())
            })
        });

        let subtree_bcubed = adj.get(node).map_or(A::zero(), |children| {
            children.iter().fold(A::zero(), |acc, c| {
                acc + *bcubed.get(c).unwrap_or(&A::zero())
            })
        });

        stability.entry(*node).and_modify(|node_stability| {
            let node_bcubed = bcubed.entry(*node).or_insert(A::zero());
            // ties are broken by stability
            if *node_bcubed > subtree_bcubed
                || (*node_bcubed == subtree_bcubed && *node_stability >= subtree_stability)
            {
                clusters[*node] = Some(*node);
            }
            *node_bcubed = node_bcubed.max(subtree_bcubed);
            *node_stability = node_stability.max(subtree_stability);
        });
    }

    // now tow-down pass to assign the clusters
    for node in nodes {
        if let Some(cluster) = clusters[node] {
            let children = adj.get(&node).expect("corrupted adjacency dictionary");
            for child in children {
                clusters[*child] = Some(cluster);
            }
        }
    }

    let num_events = condensed_tree
        .iter()
        .min_by_key(|v| v.0)
        .expect("no minimum parent available")
        .0;

    let mut res_clusters: HashMap<_, Vec<_>> = HashMap::new();
    let mut outliers = vec![];
    for (point, cluster) in clusters.iter().enumerate().take(num_events) {
        if let Some(cluster) = cluster {
            let c = res_clusters.entry(*cluster).or_default();
            c.push(point);
        } else {
            outliers.push(point);
        }
    }
    (res_clusters, outliers)
}

// GLOSH: Global-Local Outlier Score from Hierarchies
// Reference: https://dl.acm.org/doi/10.1145/2733381
//
// Given the following hierarchy (min_cluster_size = 3),
//               Root
//              /    \
//             A     ...
// eps_x ->   / \
//           x   A
//              / \
//             y   A
//                /|\   <- eps_A: A is still a cluster w.r.t. min_cluster_size at this level
//               a b c
//
// To compute the outlier score of point x, we need:
//    - eps_x: eps that x joins to cluster A (A is the first cluster that x joins to)
//    - eps_A: lowest eps that A or any of A's child clusters survives w.r.t. min_cluster_size.
// Then, the outlier score of x is defined as:
//    score(x) = 1 - eps_A / eps_x
//
// Since we are working with density lambda values (where lambda = 1/eps):
//    lambda_x = 1 / eps_x
//    lambda_A = 1 / eps_A
//    score(x) = 1 - lambda_x / lambda_A
fn glosh<A: FloatCore>(
    condensed_mst: &[(usize, usize, A, usize)],
    min_cluster_size: usize,
) -> Vec<A> {
    let deaths = max_lambdas(condensed_mst, min_cluster_size);

    // min_parent gives the number of events in the hierarchy
    let num_events = condensed_mst
        .iter()
        .map(|(parent, _, _, _)| *parent)
        .min()
        .map_or(0, |min_parent| min_parent);

    let mut scores = vec![A::zero(); num_events];
    for (parent, child, lambda, _) in condensed_mst {
        if *child >= num_events {
            continue;
        }
        let lambda_max = deaths[*parent];
        if lambda_max == A::zero() {
            scores[*child] = A::zero();
        } else {
            scores[*child] = (lambda_max - *lambda) / lambda_max;
        }
    }
    scores
}

// Return the maximum lambda value (min eps) for each cluster C such that
// the cluster or any of its child clusters has at least min_cluster_size points.
fn max_lambdas<A: FloatCore>(
    condensed_mst: &[(usize, usize, A, usize)],
    min_cluster_size: usize,
) -> Vec<A> {
    let num_clusters = condensed_mst
        .iter()
        .map(|(parent, child, _, _)| parent.max(child))
        .max()
        .expect("empty condensed_mst");

    // bottom-up traverse the hierarchy to keep track of the maximum lambda values
    // (same with the reverse order iteration on the condensed_mst)
    let mut parent_sizes: Vec<usize> = vec![0; num_clusters + 1];
    let mut deaths_arr: Vec<A> = vec![A::zero(); num_clusters + 1];
    for (parent, child, lambda, child_size) in condensed_mst.iter().rev() {
        parent_sizes[*parent] += *child_size;
        if parent_sizes[*parent] >= min_cluster_size {
            deaths_arr[*parent] = deaths_arr[*parent].max(*lambda);
        }
        if *child_size >= min_cluster_size {
            deaths_arr[*parent] = deaths_arr[*parent].max(deaths_arr[*child]);
        }
    }
    deaths_arr
}

mod test {
    /// Verifies that Boruvka and Prim's algorithms compute consistent MSTs.
    #[test]
    fn boruvka_prim_mst_consistency() {
        use ndarray::Array2;
        use petal_neighbors::distance::Euclidean;
        use petal_neighbors::BallTree;

        use crate::mst::{mst_linkage, Boruvka};

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
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            let val = (seed >> 33) as f64 / (1u64 << 31) as f64 * 20.0 - 10.0;
            data_vec.push(val);
        }

        let data: Array2<f64> =
            Array2::from_shape_vec((n_points, n_dims), data_vec).expect("valid shape");

        let metric = Euclidean::default();

        // Compute MST using Boruvka
        let db_boruvka = BallTree::new(data.view(), metric.clone()).expect("non-empty array");
        let boruvka = Boruvka::new(db_boruvka, min_samples);
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
        let mst_prim = mst_linkage(data.view(), &metric, core_distances.view(), 1.0);
        let weight_prim: f64 = mst_prim.iter().map(|(_, _, w)| *w).sum();

        // Both algorithms should produce MSTs with the same total weight.
        // The MST is unique when all edge weights are distinct, but even with
        // ties, the total weight should be identical.
        assert!(
            (weight_boruvka - weight_prim).abs() < 1e-10,
            "MST weights differ: Boruvka={}, Prim={}",
            weight_boruvka,
            weight_prim
        );
    }

    #[test]
    fn hdbscan32() {
        use ndarray::{array, Array2};
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data: Array2<f32> = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (clusters, outliers, _) = hdbscan.fit(&data, None);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn hdbscan64() {
        use ndarray::{array, Array2};
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data: Array2<f64> = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 2,
            min_cluster_size: 2,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (clusters, outliers, _) = hdbscan.fit(&data, None);
        assert_eq!(clusters.len(), 2);
        assert_eq!(
            outliers.len(),
            data.nrows() - clusters.values().fold(0, |acc, v| acc + v.len())
        );
    }

    #[test]
    fn outlier_scores() {
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data = array![
            // Cluster A (formed at eps = √2)
            [2., 9.],
            [3., 9.],
            [2., 8.],
            [3., 8.],
            [2., 7.],
            [3., 7.],
            [1., 8.],
            [4., 8.],
            // Cluster B (formed at eps = √8)
            [7., 9.],
            [7., 8.],
            [8., 8.],
            [8., 7.],
            [9., 7.],
            // Cluster C (formed at eps = 2)
            [6., 3.],
            [5., 2.],
            [6., 2.],
            [7., 2.],
            [6., 1.],
            // Outliers:
            [8., 4.], // outlier1 (joins the root cluster at eps = 3.0)
            [3., 3.], // outlier2 (joins the root cluster at eps = √13)
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 5,
            min_cluster_size: 5,
            metric: Euclidean::default(),
            boruvka: true,
        };
        let (_, _, outlier_scores) = hdbscan.fit(&data, None);

        // Outlier1 joins the root cluster at:
        //      eps_outlier1 = 3.0
        // The lowest eps that the root or any of its child clusters survive w.r.t. min_cluster_size = 5 is:
        //      eps_Root = √2
        // Then the outlier score of outlier1 is:
        //      glosh(outlier1) =  1 - √2 / 3.0 = 0.53
        let expected = 1.0 - 2.0_f64.sqrt() / 3.0_f64;
        let actual = outlier_scores[18];
        assert!(
            (actual - expected).abs() < f64::EPSILON,
            "Expected: {}, got: {}",
            expected,
            actual
        );

        // Outlier2 joins the root cluster at:
        //      eps_outlier2 = √13
        // The lowest eps that the root or any of its child clusters survive w.r.t. min_cluster_size = 5 is:
        //      eps_root = √2
        // Then the outlier score of outlier2 is:
        //      glosh(outlier2) =  1 - √2 / √13 = 0.61
        let expected = 1.0 - 2.0_f64.sqrt() / 13.0_f64.sqrt();
        let actual = outlier_scores[19];
        assert!(
            (actual - expected).abs() < f64::EPSILON,
            "Expected: {}, got: {}",
            expected,
            actual
        );
    }

    #[test]
    fn partial_labels() {
        use std::collections::HashMap;

        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        use crate::Fit;

        let data = array![
            // Group 1 (formed at eps = √2)
            [1., 9.],
            [2., 9.],
            [1., 8.],
            [2., 8.],
            [3., 7.],
            // Group 2 (formed at eps = √2)
            [5., 4.],
            [6., 4.],
            [5., 3.],
            [6., 3.],
            // Group 3 (formed at eps = √2)
            [8., 3.],
            [9., 3.],
            [8., 2.],
            [9., 2.],
            [8., 1.],
            [9., 1.],
            // noise (joins the root cluster at eps = √37)
            [7., 8.],
        ];
        let mut hdbscan = super::HDbscan {
            min_samples: 4,
            min_cluster_size: 4,
            metric: Euclidean::default(),
            boruvka: false,
            ..Default::default()
        };

        // Unsupervised clusters
        let (clusters, noise, _) = hdbscan.fit(&data, None);
        assert_eq!(clusters.len(), 2); // 2 clusters found
        assert_eq!(noise, [15]); // 1 outlier found
        let c1 = clusters.keys().find(|k| clusters[k].contains(&0)).unwrap();
        assert_eq!(clusters[c1], [0, 1, 2, 3, 4]);
        let c2 = clusters.keys().find(|k| clusters[k].contains(&5)).unwrap();
        assert_eq!(clusters[c2], [5, 6, 7, 8, 9, 10, 11, 12, 13, 14]);
        assert_eq!(noise, [15]);

        // Empty partial labels (should return the same result as unsupervised clustering)
        let partial_labels: HashMap<usize, Vec<usize>> = HashMap::new();
        let (answer, noise, _) = hdbscan.fit(&data, Some(&partial_labels));
        assert_eq!(answer, clusters);
        assert_eq!(noise, [15]);

        // Semi-supervised clustering
        let mut partial_labels: HashMap<usize, Vec<usize>> = HashMap::new();
        partial_labels.insert(0, vec![0]);
        partial_labels.insert(1, vec![3, 4]);
        partial_labels.insert(2, vec![6]);
        partial_labels.insert(3, vec![11]);
        let (clusters, noise, _) = hdbscan.fit(&data, Some(&partial_labels));
        assert_eq!(clusters.len(), 3); // 3 clusters found
        assert_eq!(noise, [15]); // 1 outlier found
        let c1 = clusters.keys().find(|k| clusters[k].contains(&0)).unwrap();
        assert_eq!(clusters[c1], [0, 1, 2, 3, 4]);
        let c2 = clusters.keys().find(|k| clusters[k].contains(&5)).unwrap();
        assert_eq!(clusters[c2], [5, 6, 7, 8]);
        let c3 = clusters.keys().find(|k| clusters[k].contains(&9)).unwrap();
        assert_eq!(clusters[c3], [9, 10, 11, 12, 13, 14]);
    }

    #[test]
    fn label() {
        let mst = vec![
            (0, 1, 4.),
            (2, 3, 4.),
            (4, 5, 4.),
            (1, 2, 7.), // <-- this (having eps = 7.0)
            (3, 4, 7.), // <-- and this (also with eps = 7.0) should have the same parent label
            (5, 6, 8.),
        ];
        // Resulting labels should be:
        //            11
        //           /  \        <-- eps = 8.0
        //          10   6
        //         / | \         <-- eps = 7.0
        //        7  8  9
        //       /|  |\  |\      <-- eps = 4.0
        //      0 1  2 3 4 5
        let labeled_mst = super::label(&mst);
        assert_eq!(
            labeled_mst,
            vec![
                (7, 0, 4., 1),
                (7, 1, 4., 1),
                (8, 2, 4., 1),
                (8, 3, 4., 1),
                (9, 4, 4., 1),
                (9, 5, 4., 1),
                (10, 7, 7., 2),
                (10, 8, 7., 2),
                (10, 9, 7., 2),
                (11, 10, 8., 6),
                (11, 6, 8., 1),
            ]
        );
    }

    #[test]
    fn get_stability() {
        use std::collections::HashMap;

        use ndarray::arr1;

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

    #[test]
    fn get_bcubed() {
        use std::collections::HashMap;

        use ndarray::arr1;

        let condensed = arr1(&[
            (8, 9, 1. / 10., 4),
            (8, 10, 1. / 10., 4),
            (9, 0, 1. / 6., 1),
            (9, 1, 1. / 7., 1),
            (9, 2, 1. / 7., 1),
            (9, 3, 1. / 6., 1),
            (10, 4, 1. / 7., 1),
            (10, 5, 1. / 6., 1),
            (10, 6, 1. / 9., 1),
            (10, 7, 1. / 9., 1),
        ]);
        let mut partial_labels = HashMap::new();
        partial_labels.insert(0, vec![0, 1, 4]);
        partial_labels.insert(1, vec![5]);
        partial_labels.insert(2, vec![7]);
        let bcubed_map: HashMap<usize, f64> = super::get_bcubed(&condensed.view(), &partial_labels);
        assert_eq!(bcubed_map.len(), 3);
        assert_eq!(bcubed_map[&8], 0.0);
        assert!((bcubed_map[&9] - 8. / 25.).abs() < f64::EPSILON);
        assert!((bcubed_map[&10] - 4. / 15.).abs() < f64::EPSILON);
    }
}
