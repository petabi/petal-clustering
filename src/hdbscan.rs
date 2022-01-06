use ndarray::{ArrayBase, Data, Ix2};
use num_traits::{Float, FromPrimitive};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ops::{AddAssign, DivAssign};

use super::Fit;
use petal_neighbors::distance::Metric;

#[derive(Debug, Deserialize, Serialize)]
pub struct HDbscan<A, M> {
    /// The radius of a neighborhood.
    pub eps: A,

    /// The minimum number of points required to form a dense region.
    pub min_samples: usize,
    pub metric: M,
}

impl<S, A, M> Fit<ArrayBase<S, Ix2>, (HashMap<usize, Vec<usize>>, Vec<usize>)> for HDbscan<A, M>
where
    A: AddAssign + DivAssign + Float + FromPrimitive + Sync,
    S: Data<Elem = A>,
    M: Metric<A> + Clone + Sync,
{
    fn fit(&mut self, _input: &ArrayBase<S, Ix2>) -> (HashMap<usize, Vec<usize>>, Vec<usize>) {
        (HashMap::new(), Vec::new())
    }
}
