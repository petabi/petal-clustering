use ndarray::ArrayView2;
use petal_neighbors::BallTree;
use std::collections::{HashMap, HashSet};

use super::Fit;

pub struct Dbscan {
    pub eps: f64,
    pub min_cluster_size: usize,
}

impl Default for Dbscan {
    fn default() -> Dbscan {
        Dbscan {
            eps: 0.5,
            min_cluster_size: 5,
        }
    }
}

impl Dbscan {
    pub fn new(eps: f64, min_cluster_size: usize) -> Self {
        Dbscan {
            eps,
            min_cluster_size,
        }
    }
}

impl<'a> Fit<'a> for Dbscan {
    type Input = ArrayView2<'a, f64>;
    type Output = (HashMap<usize, Vec<usize>>, Vec<usize>);

    fn fit(&mut self, input: Self::Input) -> Self::Output {
        // BallTree does not accept an empty input.
        if input.is_empty() {
            return (HashMap::new(), Vec::new());
        }

        let db = BallTree::new(input);
        let neighborhoods: Vec<_> = input
            .genrows()
            .into_iter()
            .map(|p| {
                db.query_radius(&p, self.eps)
                    .into_iter()
                    .collect::<Vec<_>>()
            })
            .collect();
        let mut visited = vec![false; input.rows()];
        let mut clusters = HashMap::new();

        for (idx, neighbors) in neighborhoods.iter().enumerate() {
            if visited[idx] || neighbors.len() < self.min_cluster_size {
                continue;
            }
            visited[idx] = true;

            let cid = clusters.len();
            clusters.entry(cid).or_insert_with(|| vec![idx]);
            self.expand_cluster(
                &db,
                &input,
                idx,
                cid,
                &neighborhoods,
                &mut visited,
                &mut clusters,
            );
        }

        let in_cluster: HashSet<usize> = clusters.values().flatten().cloned().collect();
        let outliers = (0..input.rows())
            .filter(|x| !in_cluster.contains(x))
            .collect();

        (clusters, outliers)
    }
}

impl Dbscan {
    fn expand_cluster(
        &mut self,
        db: &BallTree,
        input: &ArrayView2<f64>,
        core_idx: usize,
        cid: usize,
        neighborhoods: &[Vec<usize>],
        visited: &mut [bool],
        clusters: &mut HashMap<usize, Vec<usize>>,
    ) {
        for &neighbor in &neighborhoods[core_idx] {
            if visited[neighbor] {
                continue;
            } else {
                let cluster = clusters.entry(cid).or_insert_with(|| vec![]);
                cluster.push(neighbor);
                visited[neighbor] = true;
            }
            let neighbors = &neighborhoods[neighbor];
            if neighbors.len() < self.min_cluster_size {
                continue;
            }
            self.expand_cluster(db, input, neighbor, cid, &neighborhoods, visited, clusters);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use ndarray::aview2;

    fn comparison(
        data: Vec<f64>,
        dim: (usize, usize),
        eps: f64,
        min_cluster_size: usize,
    ) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        use rusty_machine::learning::dbscan::DBSCAN;
        use rusty_machine::learning::UnSupModel;
        use rusty_machine::linalg::Matrix;

        let (rows, cols) = dim;
        let inputs = Matrix::new(rows, cols, data);

        let mut model = DBSCAN::new(eps, min_cluster_size);
        model.train(&inputs).unwrap();
        let clustering = model.clusters().unwrap();

        let mut clusters = HashMap::new();
        let mut outliers = vec![];

        for (idx, &cid) in clustering.iter().enumerate() {
            match cid {
                Some(cid) => {
                    let cluster = clusters.entry(cid).or_insert_with(|| vec![]);
                    cluster.push(idx);
                }
                None => outliers.push(idx),
            }
        }
        (clusters, outliers)
    }

    #[test]
    fn dbscan() {
        let data = vec![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let input = aview2(&data);

        let mut model = Dbscan::new(0.5, 2);
        let (mut clusters, mut outliers) = model.fit(input);
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }
        let answer = comparison(data.iter().flatten().cloned().collect(), (6, 2), 0.5, 2);

        assert_eq!(answer.0, clusters);
        assert_eq!(answer.1, outliers);
    }

    #[test]
    fn dbscan_core_samples() {
        let data = vec![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Dbscan::new(1.01, 1);
        let (clusters, outliers) = model.fit(aview2(&data));
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Dbscan::new(0.5, 2);
        let (clusters, outliers) = model.fit(input);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }
}
