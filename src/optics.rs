use ndarray::{Array, ArrayBase, ArrayView1, Data, Ix2};
use num_traits::{Float, FromPrimitive};
use petal_neighbors::BallTree;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use super::Fit;

/// OPTICS (ordering points to identify the clustering structure) clustering
/// algorithm.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use petal_clustering::{Optics, Fit};
///
/// let points = array![[1.0, 2.0], [2.0, 5.0], [3.0, 6.0], [8.0, 7.0], [8.0, 8.0], [7.0, 3.0]];
/// let clustering = Optics::new(4.5, 2).fit(&points);
///
/// assert_eq!(clustering.0.len(), 2);        // two clusters found
/// assert_eq!(clustering.0[&0], [0, 1, 2]);  // the first three points in Cluster 0
/// assert_eq!(clustering.0[&1], [3, 4, 5]);  // the rest in Cluster 1
/// ```
#[derive(Debug, Deserialize, Serialize)]
pub struct Optics<A> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,

    ordered: Vec<usize>,
    reacheability: Vec<A>,
    neighborhoods: Vec<Neighborhood<A>>,
}

impl<A> Default for Optics<A>
where
    A: Float,
{
    #[must_use]
    fn default() -> Self {
        Self {
            eps: A::from(0.5_f32).expect("valid float"),
            min_samples: 5,
            ordered: vec![],
            reacheability: vec![],
            neighborhoods: vec![],
        }
    }
}

impl<A> Optics<A>
where
    A: Float,
{
    #[must_use]
    pub fn new(eps: A, min_samples: usize) -> Self {
        Self {
            eps,
            min_samples,
            ordered: vec![],
            reacheability: vec![],
            neighborhoods: vec![],
        }
    }

    #[must_use]
    pub fn extract_clusters_and_outliers(
        &self,
        eps: A,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        let mut outliers = vec![];
        let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

        for &id in &self.ordered {
            if self.reacheability[id].is_normal() && self.reacheability[id] <= eps {
                if clusters.is_empty() {
                    outliers.push(id);
                } else {
                    let v = clusters
                        .get_mut(&(clusters.len() - 1))
                        .expect("cluster map crashed");
                    v.push(id);
                }
            } else {
                let n = &self.neighborhoods[id];
                if n.neighbors.len() >= self.min_samples && n.core_distance <= eps {
                    clusters.entry(clusters.len()).or_insert_with(|| vec![id]);
                } else {
                    outliers.push(id);
                }
            }
        }
        (clusters, outliers)
    }
}

impl<S, A> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for Optics<A>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Send + Sync,
    S: Data<Elem = A> + Sync,
{
    fn fit(&mut self, input: &ArrayBase<S, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        if input.is_empty() {
            return (HashMap::new(), vec![]);
        }

        self.neighborhoods = if input.is_standard_layout() {
            build_neighborhoods(input, self.eps)
        } else {
            let input = Array::from_shape_vec(input.raw_dim(), input.iter().copied().collect())
                .expect("valid shape");
            build_neighborhoods(&input, self.eps)
        };
        let mut visited = vec![false; input.nrows()];
        self.ordered = Vec::with_capacity(input.nrows());
        self.reacheability = vec![A::nan(); input.nrows()];
        for (idx, n) in self.neighborhoods.iter().enumerate() {
            if visited[idx] || n.neighbors.len() < self.min_samples {
                continue;
            }
            process(
                idx,
                input,
                self.min_samples,
                &self.neighborhoods,
                &mut self.ordered,
                &mut self.reacheability,
                &mut visited,
            );
        }
        self.extract_clusters_and_outliers(self.eps)
    }
}

fn process<S, A>(
    idx: usize,
    input: &ArrayBase<S, Ix2>,
    min_samples: usize,
    neighborhoods: &[Neighborhood<A>],
    ordered: &mut Vec<usize>,
    reacheability: &mut Vec<A>,
    visited: &mut [bool],
) where
    A: Float,
    S: Data<Elem = A>,
{
    let mut to_visit = vec![idx];
    while let Some(cur) = to_visit.pop() {
        if visited[cur] {
            continue;
        }
        visited[cur] = true;
        ordered.push(cur);
        if neighborhoods[cur].neighbors.len() < min_samples {
            continue;
        }
        let mut seeds = vec![];
        update(
            cur,
            &neighborhoods[cur],
            input,
            &visited,
            &mut seeds,
            reacheability,
        );
        while let Some(s) = seeds.pop() {
            if visited[s] {
                continue;
            }
            visited[s] = true;
            ordered.push(s);
            let n = &neighborhoods[s];
            if n.neighbors.len() < min_samples {
                continue;
            }
            update(
                s,
                &neighborhoods[s],
                input,
                &visited,
                &mut seeds,
                reacheability,
            );
        }
    }
}

fn update<S, A>(
    id: usize,
    neighborhood: &Neighborhood<A>,
    input: &ArrayBase<S, Ix2>,
    visited: &[bool],
    seeds: &mut Vec<usize>,
    reacheability: &mut [A],
) where
    A: Float,
    S: Data<Elem = A>,
{
    for &o in &neighborhood.neighbors {
        if visited[o] {
            continue;
        }
        let reachdist = reacheability_distance(o, id, input, neighborhood);
        if !reacheability[o].is_normal() {
            reacheability[o] = reachdist;
            seeds.push(o);
        } else if reacheability[o].lt(&reachdist) {
            reacheability[o] = reachdist;
        }
    }
    seeds.sort_unstable_by(|a, b| {
        reacheability[*a]
            .partial_cmp(&reacheability[*b])
            .unwrap()
            .reverse()
    });
}

#[derive(Debug, Deserialize, Serialize)]
struct Neighborhood<A> {
    pub neighbors: Vec<usize>,
    pub core_distance: A,
}

fn build_neighborhoods<S, A>(input: &ArrayBase<S, Ix2>, eps: A) -> Vec<Neighborhood<A>>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Send + Sync,
    S: Data<Elem = A>,
{
    if input.nrows() == 0 {
        return Vec::new();
    }
    let rows: Vec<_> = input.rows().into_iter().collect();
    let db = BallTree::euclidean(input.view()).expect("non-empty array");
    rows.into_par_iter()
        .map(|p| {
            let neighbors = db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>();
            let core_distance = if neighbors.len() > 1 {
                db.query(&p, 2).1[1]
            } else {
                A::zero()
            };
            Neighborhood {
                neighbors,
                core_distance,
            }
        })
        .collect()
}

fn distance<A>(a: &ArrayView1<A>, b: &ArrayView1<A>) -> A
where
    A: Float,
{
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

fn reacheability_distance<S, A>(
    o: usize,
    p: usize,
    input: &ArrayBase<S, Ix2>,
    neighbors: &Neighborhood<A>,
) -> A
where
    A: Float,
    S: Data<Elem = A>,
{
    let dist = distance(&input.row(o), &input.row(p));
    if dist.gt(&neighbors.core_distance) {
        dist
    } else {
        neighbors.core_distance
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use maplit::hashmap;
    use ndarray::{array, aview2};

    #[test]
    fn default() {
        let optics = Optics::<f32>::default();
        assert_eq!(optics.eps, 0.5);
        assert_eq!(optics.min_samples, 5);
    }

    #[test]
    fn optics() {
        let data = array![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];

        let mut model = Optics::new(0.5, 2);
        let (mut clusters, mut outliers) = model.fit(&data);
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }

        assert_eq!(hashmap! {0 => vec![0, 1, 2, 3], 1 => vec![4, 5]}, clusters);
        assert_eq!(Vec::<usize>::new(), outliers);
    }

    #[test]
    fn core_samples() {
        let data = array![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Optics::new(1.01, 1);
        let (clusters, outliers) = model.fit(&data);
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Optics::new(0.5, 2);
        let (clusters, outliers) = model.fit(&input);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }
}
