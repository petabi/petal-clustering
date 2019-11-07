use ndarray::{ArrayView1, ArrayView2};
use petal_neighbors::{distance, BallTree};
use rayon::prelude::*;
use std::collections::HashMap;

use super::Fit;

pub struct Optics {
    pub eps: f64,
    pub min_samples: usize,
}

impl Default for Optics {
    fn default() -> Self {
        Self {
            eps: 0.5,
            min_samples: 5,
        }
    }
}

impl Optics {
    pub fn new(eps: f64, min_samples: usize) -> Self {
        Self { eps, min_samples }
    }
}

impl<'a> Fit<'a> for Optics {
    type Input = ArrayView2<'a, f64>;
    type Output = (HashMap<usize, Vec<usize>>, Vec<usize>);

    fn fit(&mut self, input: Self::Input) -> Self::Output {
        if input.is_empty() {
            return (HashMap::new(), vec![]);
        }
        let neighborhoods = build_neighborhoods(&input, self.eps);
        let mut visited = vec![false; input.nrows()];
        let mut ordered: Vec<usize> = Vec::with_capacity(input.nrows());
        let mut reacheability = vec![std::f64::NAN; input.nrows()];
        for (idx, n) in neighborhoods.iter().enumerate() {
            if visited[idx] || n.neighbors.len() < self.min_samples {
                continue;
            }
            process(
                idx,
                &input,
                self.min_samples,
                &neighborhoods,
                &mut ordered,
                &mut reacheability,
                &mut visited,
            );
        }
        extract_clusters_and_outliers(
            &ordered,
            &reacheability,
            &neighborhoods,
            self.eps,
            self.min_samples,
        )
    }
}

fn extract_clusters_and_outliers(
    ordered: &[usize],
    reacheability: &[f64],
    neighborhoods: &[Neighborhood],
    eps: f64,
    min_samples: usize,
) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
    let mut outliers = vec![];
    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();

    for &id in ordered {
        if reacheability[id].is_normal() && reacheability[id] <= eps {
            if clusters.is_empty() {
                outliers.push(id);
            } else {
                let v = clusters
                    .get_mut(&(clusters.len() - 1))
                    .expect("cluster map crashed");
                v.push(id);
            }
        } else {
            let n = &neighborhoods[id];
            if n.neighbors.len() >= min_samples && n.core_distance <= eps {
                clusters.entry(clusters.len()).or_insert_with(|| vec![id]);
            } else {
                outliers.push(id);
            }
        }
    }
    (clusters, outliers)
}

fn process<'a>(
    idx: usize,
    input: &ArrayView2<'a, f64>,
    min_samples: usize,
    neighborhoods: &[Neighborhood],
    ordered: &mut Vec<usize>,
    reacheability: &mut Vec<f64>,
    visited: &mut [bool],
) {
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

fn update<'a>(
    id: usize,
    neighborhood: &Neighborhood,
    input: &ArrayView2<'a, f64>,
    visited: &[bool],
    seeds: &mut Vec<usize>,
    reacheability: &mut [f64],
) {
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

#[derive(Debug)]
struct Neighborhood {
    pub neighbors: Vec<usize>,
    pub core_distance: f64,
}

fn build_neighborhoods<'a>(input: &ArrayView2<'a, f64>, eps: f64) -> Vec<Neighborhood> {
    let rows: Vec<_> = input.genrows().into_iter().collect();
    let db = BallTree::with_metric(*input, distance::EUCLIDEAN);
    rows.into_par_iter()
        .map(|p| {
            let neighbors = db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>();
            let core_distance = if neighbors.len() > 1 {
                let ns = db.query(&p, 2);
                if ns[0].distance.gt(&ns[1].distance) {
                    ns[0].distance
                } else {
                    ns[1].distance
                }
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

fn reacheability_distance<'a>(
    o: usize,
    p: usize,
    input: &ArrayView2<'a, f64>,
    neighbors: &Neighborhood,
) -> f64 {
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
    fn optics() {
        let data = vec![
            [1.0, 2.0],
            [1.1, 2.2],
            [0.9, 1.9],
            [1.0, 2.1],
            [-2.0, 3.0],
            [-2.2, 3.1],
        ];
        let input = aview2(&data);

        let mut model = Optics::new(0.5, 2);
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
    fn core_samples() {
        let data = vec![[0.], [2.], [3.], [4.], [6.], [8.], [10.]];
        let mut model = Optics::new(1.01, 1);
        let (clusters, outliers) = model.fit(aview2(&data));
        assert_eq!(clusters.len(), 5); // {0: [0], 1: [1, 2, 3], 2: [4], 3: [5], 4: [6]}
        assert!(outliers.is_empty());
    }

    #[test]
    fn fit_empty() {
        let data: Vec<[f64; 8]> = vec![];
        let input = aview2(&data);

        let mut model = Optics::new(0.5, 2);
        let (clusters, outliers) = model.fit(input);
        assert!(clusters.is_empty());
        assert!(outliers.is_empty());
    }
}
