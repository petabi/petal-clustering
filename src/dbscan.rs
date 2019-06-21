use ndarray::ArrayView2;
use petal_neighbors::{BallTree, Neighbor};
use std::collections::{HashMap, HashSet};

use super::Fit;

pub struct DBSCAN {
    pub eps: f64,
    pub min_cluster_size: usize,
}

impl Default for DBSCAN {
    fn default() -> DBSCAN {
        DBSCAN {
            eps: 0.5,
            min_cluster_size: 5,
        }
    }
}

impl DBSCAN {
    pub fn new(eps: f64, min_cluster_size: usize) -> Self {
        DBSCAN {
            eps,
            min_cluster_size,
        }
    }
}

impl<'a> Fit<'a> for DBSCAN {
    type Input = ArrayView2<'a, f64>;
    type Output = (HashMap<usize, Vec<usize>>, Vec<usize>);

    fn fit(&mut self, input: Self::Input) -> Self::Output {
        let db = BallTree::new(input);
        let mut visited = vec![false; input.rows()];
        let mut clusters = HashMap::new();

        for (idx, point) in (0..input.rows()).zip(input.genrows()) {
            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            let neighbors = db.query_radius(&point, self.eps);
            if neighbors.len() < self.min_cluster_size {
                continue;
            }
            let cid = clusters.len();
            clusters.entry(cid).or_insert_with(|| vec![idx]);
            self.expand_cluster(&db, &input, cid, &neighbors, &mut visited, &mut clusters);
        }

        let in_cluster: HashSet<usize> = clusters.values().flatten().cloned().collect();
        let outliers = (0..input.rows())
            .filter(|x| !in_cluster.contains(x))
            .collect();

        (clusters, outliers)
    }
}

impl DBSCAN {
    fn expand_cluster(
        &mut self,
        db: &BallTree,
        input: &ArrayView2<f64>,
        cid: usize,
        neighbors: &[Neighbor],
        visited: &mut [bool],
        clusters: &mut HashMap<usize, Vec<usize>>,
    ) {
        for neighbor in neighbors {
            let idx = neighbor.idx;

            if visited[idx] {
                continue;
            }
            visited[idx] = true;
            let point = input.row(idx);
            let neighbors = db.query_radius(&point, self.eps);
            if neighbors.len() < self.min_cluster_size {
                continue;
            }
            {
                let cluster = clusters.entry(cid).or_insert_with(|| vec![]);
                cluster.push(idx);
            }
            self.expand_cluster(db, input, cid, &neighbors, visited, clusters);
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

        let mut model = DBSCAN::new(0.5, 2);
        let (mut clusters, mut outliers) = model.fit(input);
        outliers.sort_unstable();
        for (_, v) in clusters.iter_mut() {
            v.sort_unstable();
        }
        let answer = comparison(data.iter().flatten().cloned().collect(), (6, 2), 0.5, 2);

        assert_eq!(answer.0, clusters);
        assert_eq!(answer.1, outliers);
    }
}
