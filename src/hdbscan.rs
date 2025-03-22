use std::collections::{HashMap, HashSet};
use std::fmt::Debug;
use std::ops::{AddAssign, DivAssign, Sub};

use ndarray::{Array1, ArrayBase, ArrayView1, Data, Ix2};
use num_traits::{float::FloatCore, FromPrimitive};
use petal_neighbors::distance::{Euclidean, Metric};
use petal_neighbors::BallTree;
use serde::{Deserialize, Serialize};

use super::Fit;
use crate::mst::{bfs_tree, condense_mst, mst_linkage, Boruvka};
use crate::union_find::{TreeUnionFind, UnionFind};

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
    #[must_use]
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
        let sorted_mst = Array1::from_vec(mst);
        let labeled = label(sorted_mst);
        let condensed = condense_mst(labeled.view(), self.min_cluster_size);
        let outlier_scores = glosh(&condensed, self.min_cluster_size);
        let (clusters, outliers) =
            find_clusters(&Array1::from_vec(condensed).view(), partial_labels);
        (clusters, outliers, outlier_scores)
    }
}

fn label<A: FloatCore>(mst: Array1<(usize, usize, A)>) -> Array1<(usize, usize, A, usize)> {
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

    let max_parent = condensed_tree
        .iter()
        .map(|(parent, child, _, _)| parent.max(child))
        .max()
        .expect("empty condensed_mst");

    // bottom-up traverse the hierarchy to keep track of the counts of the labelled points
    // (same with the reverse order iteration on the condensed_mst)
    let mut label_map: HashMap<usize, HashMap<usize, A>> = HashMap::new();
    let mut num_labels: Vec<A> = vec![A::zero(); max_parent + 1];
    let mut bcubed: Vec<A> = vec![A::zero(); max_parent + 1];
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
            if child_num_labelled == A::zero() {
                continue;
            }

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
                acc + *stability.get(c).expect("corrupted stability dictionary")
            } else {
                acc
            }
        });

        let subtree_bcubed = tree.iter().fold(A::zero(), |acc, (p, c)| {
            if *p == node {
                acc + *bcubed.get(c).expect("corrupted bcubed dictionary")
            } else {
                acc
            }
        });

        stability.entry(node).and_modify(|node_stability| {
            let node_bcubed = bcubed.entry(node).or_insert(A::zero());
            // ties are broken by stability
            if *node_bcubed < subtree_bcubed
                || (*node_bcubed == subtree_bcubed && *node_stability < subtree_stability)
            {
                clusters.remove(&node);
                *node_bcubed = subtree_bcubed;
                *node_stability = subtree_stability.max(*node_stability);
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
    let largest_cluster_id = condensed_mst
        .iter()
        .map(|(parent, child, _, _)| parent.max(child))
        .max()
        .expect("empty condensed_mst");

    // bottom-up traverse the hierarchy to keep track of the maximum lambda values
    // (same with the reverse order iteration on the condensed_mst)
    let mut parent_sizes: Vec<usize> = vec![0; largest_cluster_id + 1];
    let mut deaths_arr: Vec<A> = vec![A::zero(); largest_cluster_id + 1];
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
        use crate::Fit;
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;

        let data = array![
            // cluster1:
            [1.0, 1.0],
            [1.0, 2.0],
            [2.0, 1.0],
            [2.0, 2.0],
            // cluster2:
            [4.0, 1.0],
            [4.0, 2.0],
            [5.0, 1.0],
            [5.0, 2.0],
            // cluster3:
            [9.0, 1.0],
            [9.0, 2.0],
            [10.0, 1.0],
            [10.0, 2.0],
            [11.0, 1.0],
            [11.0, 2.0],
            // outlier1:
            [2.0, 5.0],
            // outlier2:
            [10.0, 8.0],
        ];
        let mut hdbscan = super::HDbscan {
            alpha: 1.,
            min_samples: 4,
            min_cluster_size: 4,
            metric: Euclidean::default(),
            boruvka: false,
        };
        let (_, _, outlier_scores) = hdbscan.fit(&data, None);

        // The first 14 data objects immediately form their clusters at eps = √2
        // The outlier scores of these objects are all 0:
        //      glosh(x) = 1 - √2 / √2 = 0
        for i in 0..14 {
            assert_eq!(outlier_scores[i], 0.0);
        }

        // Outlier1 joins the cluster C = {cluster1 ∪ cluster2} at:
        //      eps_outlier1 = √13
        // The lowest eps that C or any of its child clusters survives w.r.t. min_cluster_size = 4 is:
        //      eps_C = √2 (due to cluster1 or cluster2)
        // Then the outlier score of outlier1 is:
        //      glosh(outlier1) =  1 - √2 / √13 = 0.60776772972
        assert_eq!(outlier_scores[14], 1.0 - 2.0_f64.sqrt() / 13.0_f64.sqrt());

        // Outlier2 joins the root cluster at at eps = √37
        // The lowest eps that the root cluster survives w.r.t. min_cluster_size = 4 is:
        //      eps_root = √2
        // Then the outlier score of outlier2 is:
        //      glosh(outlier2) =  1 - √2 / √37 = 0.76750472251
        assert_eq!(outlier_scores[15], 1.0 - 2.0_f64.sqrt() / 37.0_f64.sqrt());
    }

    #[test]
    fn partial_labels() {
        use crate::Fit;
        use ndarray::array;
        use petal_neighbors::distance::Euclidean;
        use std::collections::HashMap;

        let data = array![
            // Group 1 (formed at eps = √2)
            [1.0, 9.0],
            [2.0, 9.0],
            [1.0, 8.0],
            [2.0, 8.0],
            [3.0, 7.0],
            // Group 2 (formed at eps = √2)
            [5.0, 4.0],
            [6.0, 4.0],
            [5.0, 3.0],
            [6.0, 3.0],
            // Group 3 (formed at eps = √2)
            [8.0, 3.0],
            [9.0, 3.0],
            [8.0, 2.0],
            [9.0, 2.0],
            [8.0, 1.0],
            [9.0, 1.0],
            // noise (joins the root cluster at eps = √37)
            [7.0, 8.0],
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
        use ndarray::arr1;
        use std::collections::HashMap;

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
        assert!((bcubed_map[&9] - 8.0 / 25.0).abs() < f64::EPSILON);
        assert!((bcubed_map[&10] - 4.0 / 15.0).abs() < f64::EPSILON);
    }
}
