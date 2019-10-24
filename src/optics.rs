use ndarray::ArrayView2;
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
        let _neighborhoods = build_neighborhoods(&input, self.eps);
        (HashMap::new(), vec![])
    }
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
