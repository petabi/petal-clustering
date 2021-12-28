use ndarray::ArrayView1;
use ndarray_rand::rand::{rngs::StdRng, RngCore, SeedableRng};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};

/// make_a_blob: generate an isotropic Gaussian blob,
///  centered at `center` with standard deviation `std_dev`
///  blob size: `n_smaples`
/// data is returned in form of Vec<f64> (COLUMN major: `n_features` * `n_samples`)
fn make_a_blob<R: RngCore + Clone>(
    center: ArrayView1<f64>,
    n_samples: usize,
    std_dev: f64,
    seed_rng: R,
) -> Vec<f64> {
    let mut data = Vec::new();
    for c in center {
        let mut rng = StdRng::from_rng(seed_rng.clone()).unwrap();
        let norm = Normal::new(*c, std_dev).unwrap();
        data.extend(norm.sample_iter(&mut rng).take(n_samples));
    }
    data
}

/// make_centers Generate uniformly distributed `n_centers * n_features` centers
/// within the bounding box provided by `center_box`. Rng used is seeded by `seed_rng`.
/// results are returned in form of Vec<f64> (row major n_centers * n_features).
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
