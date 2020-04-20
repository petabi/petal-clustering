use ndarray::{ArrayBase, ArrayView1, Data, Ix2};
use petal_neighbors::{distance, BallTree};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

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
pub struct Optics {
    /// The radius of a neighborhood.
    pub eps: f64,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,

    ordered: Vec<usize>,
    reacheability: Vec<f64>,
    neighborhoods: Vec<Neighborhood>,
}

impl Default for Optics {
    #[must_use]
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
            ordered: vec![],
            reacheability: vec![],
            neighborhoods: vec![],
        }
    }
}

impl Optics {
    #[must_use]
    pub fn new(eps: f64, min_samples: usize) -> Self {
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
        eps: f64,
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

impl<D> Fit<ArrayBase<D, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for Optics
where
    D: Data<Elem = f64> + Sync,
{
    fn fit(&mut self, input: &ArrayBase<D, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        if input.is_empty() {
            return (HashMap::new(), vec![]);
        }
        self.neighborhoods = build_neighborhoods(&input, self.eps);
        let mut visited = vec![false; input.nrows()];
        self.ordered = Vec::with_capacity(input.nrows());
        self.reacheability = vec![std::f64::NAN; input.nrows()];
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

fn process<D>(
    idx: usize,
    input: &ArrayBase<D, Ix2>,
    min_samples: usize,
    neighborhoods: &[Neighborhood],
    ordered: &mut Vec<usize>,
    reacheability: &mut Vec<f64>,
    visited: &mut [bool],
) where
    D: Data<Elem = f64>,
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

fn update<D>(
    id: usize,
    neighborhood: &Neighborhood,
    input: &ArrayBase<D, Ix2>,
    visited: &[bool],
    seeds: &mut Vec<usize>,
    reacheability: &mut [f64],
) where
    D: Data<Elem = f64>,
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
struct Neighborhood {
    pub neighbors: Vec<usize>,
    pub core_distance: f64,
}

fn build_neighborhoods<'a, D>(input: &'a ArrayBase<D, Ix2>, eps: f64) -> Vec<Neighborhood>
where
    D: Data<Elem = f64> + Sync,
{
    let rows: Vec<_> = input.genrows().into_iter().collect();
    let db = BallTree::new(input.view(), distance::EUCLIDEAN).unwrap();
    rows.into_par_iter()
        .map(|p| {
            let neighbors = db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>();
            let core_distance = if neighbors.len() > 1 {
                db.query(&p, 2).1[1]
            } else {
                0.0
            };
            Neighborhood {
                neighbors,
                core_distance,
            }
        })
        .collect()
}

fn distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    (a - b).mapv(|x| x.powi(2)).sum().sqrt()
}

fn reacheability_distance<D>(
    o: usize,
    p: usize,
    input: &ArrayBase<D, Ix2>,
    neighbors: &Neighborhood,
) -> f64
where
    D: Data<Elem = f64>,
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
