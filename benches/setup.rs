use ndarray_rand::rand::distributions::{Distribution, Uniform};
use ndarray_rand::rand::{rngs::StdRng, RngCore, SeedableRng};

/// make_centers Generate uniformly distributed `n_centers * n_features` centers
/// within the bounding box provided by `center_box`. Rng used is seeded by `seed_rng`.
/// results are returned in form of Vec<f64>.
fn make_centers<R: RngCore>(
    n_centers: usize,
    n_features: usize,
    center_box: (f64, f64),
    seed_rng: R,
) -> Vec<f64> {
    let (low, high) = center_box;
    let mut rng = StdRng::from_rng(seed_rng).unwrap();
    let between = Uniform::new(low, high);
    between
        .sample_iter(&mut rng)
        .take(n_centers * n_features)
        .collect()
}
