use ndarray::{ArrayView1, ArrayView2};
use petal_neighbors::BallTree;
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
                &input,
                &mut ordered,
                &mut reacheability,
                &mut visited,
                idx,
                self.min_samples,
                &neighborhoods,
            );
        }
        (HashMap::new(), vec![])
    }
}

fn process<'a>(
    input: &ArrayView2<'a, f64>,
    ordered: &mut Vec<usize>,
    reacheability: &mut Vec<f64>,
    visited: &mut [bool],
    idx: usize,
    min_samples: usize,
    neighborhoods: &[Neighborhood],
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
            input,
            &neighborhoods[cur],
            cur,
            &mut seeds,
            &visited,
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
                input,
                &neighborhoods[cur],
                cur,
                &mut seeds,
                &visited,
                reacheability,
            );
        }
    }
}

fn update<'a>(
    input: &ArrayView2<'a, f64>,
    neighborhood: &Neighborhood,
    id: usize,
    seeds: &mut Vec<usize>,
    visited: &[bool],
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

struct Neighborhood {
    pub neighbors: Vec<usize>,
    pub core_distance: f64,
}

fn build_neighborhoods<'a>(input: &ArrayView2<'a, f64>, eps: f64) -> Vec<Neighborhood> {
    let rows: Vec<_> = input.genrows().into_iter().collect();
    let db = BallTree::new(*input);
    rows.into_par_iter()
        .map(|p| Neighborhood {
            neighbors: db.query_radius(&p, eps).into_iter().collect::<Vec<usize>>(),
            core_distance: db.query_one(&p).distance,
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
