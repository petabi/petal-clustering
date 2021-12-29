use ndarray::{concatenate, Array2, ArrayView, ArrayView1, Axis};
use ndarray_rand::rand::{
    rngs::{OsRng, StdRng},
    RngCore, SeedableRng,
};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};

const DEFAULT_CLUSTER_STD: f64 = 1.0;
const DEFAULT_CENTERS: usize = 3;
const DEFAULT_CENTER_BOX: (f64, f64) = (-10., 10.);

pub(crate) fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: Option<usize>,
    cluster_std: Option<f64>,
    center_box: Option<(f64, f64)>,
) -> Array2<f64> {
    let centers = centers.unwrap_or(DEFAULT_CENTERS);
    let cluster_std = cluster_std.unwrap_or(DEFAULT_CLUSTER_STD);
    let center_box = center_box.unwrap_or(DEFAULT_CENTER_BOX);

    let centers_data = make_centers(centers, n_features, center_box, OsRng);
    let samples_per_center = n_samples / centers;
    let centers = ArrayView::from_shape((centers, n_features), &centers_data).unwrap();
    let mut data = vec![];
    for center in centers.rows() {
        data.push(make_a_blob(center, samples_per_center, cluster_std, OsRng));
    }
    let blobs: Vec<_> = data
        .iter()
        .map(|blob| {
            ArrayView::from_shape((n_features, samples_per_center), blob.as_slice())
                .unwrap()
                .reversed_axes()
        })
        .collect();
    concatenate(Axis(0), blobs.as_slice()).unwrap()
}

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
