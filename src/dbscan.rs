use ndarray::{Array, ArrayBase, Data, Ix2};
use num_traits::{Float, FromPrimitive};
use petal_neighbors::BallTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::ops::{AddAssign, DivAssign};

use super::Fit;

/// DBSCAN (density-based spatial clustering of applications with noise)
/// clustering algorithm.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use petal_clustering::{Dbscan, Fit};
///
/// let points = array![[1.0, 2.0], [2.0, 2.0], [2.0, 2.3], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]];
/// let clustering = Dbscan::new(3.0, 2).fit(&points);
///
/// assert_eq!(clustering.0.len(), 2);        // two clusters found
/// assert_eq!(clustering.0[&0], [0, 1, 2]);  // the first three points in Cluster 0
/// assert_eq!(clustering.0[&1], [3, 4]);     // [8.0, 7.0] and [8.0, 8.0] in Cluster 1
/// assert_eq!(clustering.1, [5]);            // [25.0, 80.0] doesn't belong to any cluster
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct Dbscan<A> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
}

impl<A> Default for Dbscan<A>
where
    A: Float,
{
    #[must_use]
    fn default() -> Self {
        Self {
            eps: A::from(0.5_f32).expect("valid float"),
            min_samples: 5,
        }
    }
}

impl<A> Dbscan<A> {
    #[must_use]
    pub fn new(eps: A, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }
}

impl<S, A> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for Dbscan<A>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync,
    S: Data<Elem = A>,
{
    fn fit(&mut self, input: &ArrayBase<S, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        // `BallTree` does not accept an empty input.
        if input.is_empty() {
            return (HashMap::new(), Vec::new());
        }

        let neighborhoods = if input.is_standard_layout() {
            build_neighborhoods(input, self.eps)
        } else {
            let input = Array::from_shape_vec(input.raw_dim(), input.iter().cloned().collect())
                .expect("valid shape");
            build_neighborhoods(&input, self.eps)
        };
        let mut visited = vec![false; input.nrows()];
        let mut clusters = HashMap::new();
        for (idx, neighbors) in neighborhoods.iter().enumerate() {
            if visited[idx] || neighbors.len() < self.min_samples {
                continue;
            }

            let cid = clusters.len();
            let cluster = clusters.entry(cid).or_insert_with(Vec::new);
            expand_cluster(cluster, &mut visited, idx, self.min_samples, &neighborhoods);
        }

        let in_cluster: HashSet<usize> = clusters.values().flatten().cloned().collect();
        let outliers = (0..input.nrows())
            .filter(|x| !in_cluster.contains(x))
            .collect();

        (clusters, outliers)
    }
}

fn build_neighborhoods<S, A>(input: &ArrayBase<S, Ix2>, eps: A) -> Vec<Vec<usize>>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync,
    S: Data<Elem = A>,
{
    if input.nrows() == 0 {
        return Vec::new();
    }
    let rows: Vec<_> = input.genrows().into_iter().collect();
    let db = BallTree::euclidean(input.view()).expect("non-empty array");
    rows.into_par_iter()
        .map(|p| db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>())
        .collect()
}

fn expand_cluster(
    cluster: &mut Vec<usize>,
    visited: &mut [bool],
    idx: usize,
    min_samples: usize,
    neighborhoods: &[Vec<usize>],
) {
    let mut to_visit = vec![idx];
    while let Some(cur) = to_visit.pop() {
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        cluster.push(cur);
        if neighborhoods[cur].len() >= min_samples {
            to_visit.extend(neighborhoods[cur].iter().filter(|&n| !visited[*n]));
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use maplit::hashmap;
    use ndarray::{array, aview2};

    #[test]
    fn default() {
        let dbscan = Dbscan::<f32>::default();
        assert_eq!(dbscan.eps, 0.5);
        assert_eq!(dbscan.min_samples, 5);
    }

    #[test]
    fn dbscan() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];

        let mut model = Dbscan::new(0.5, 2);
        let (mut clusters, mut outliers) = model.fit(&data);
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }

        assert_eq!(hashmap! {0 => vec![0, 1, 2, 3], 1 => vec![4, 5]}, clusters);
        assert_eq!(Vec::<usize>::new(), outliers);
    }

    #[test]
    fn dbscan_core_samples() {
        let data = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Dbscan::new(1.01, 1);
        let (clusters, outliers) = model.fit(&data);
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Dbscan::new(0.5, 2);
        let (clusters, outliers) = model.fit(&input);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }
}
