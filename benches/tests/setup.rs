use ndarray::{concatenate, Array2, ArrayView, ArrayView1, Axis};
use ndarray_rand::rand::{
    rngs::{OsRng, StdRng},
    RngCore, SeedableRng,
};
use ndarray_rand::rand_distr::{Distribution, Normal, Uniform};

const DEFAULT_CLUSTER_STD: f64 = 1.0;
const DEFAULT_N_CENTERS: usize = 3;
const DEFAULT_CENTER_BOX: (f64, f64) = (-10., 10.);

pub(crate) enum CenterConfig {
    Fixed(Array2<f64>),
    Random(usize, (f64, f64)),
}

impl Default for CenterConfig {
    fn default() -> Self {
        Self::Random(DEFAULT_N_CENTERS, DEFAULT_CENTER_BOX)
    }
}

#[must_use]
pub(crate) fn make_blobs(
    n_samples: usize,
    n_features: usize,
    center_config: Option<CenterConfig>,
    cluster_std: Option<f64>,
    random_state: Option<[u8; 32]>,
) -> Array2<f64> {
    let center_config = center_config.unwrap_or_default();
    let cluster_std = cluster_std.unwrap_or(DEFAULT_CLUSTER_STD);

    let centers_data = match center_config {
        CenterConfig::Fixed(centers) => centers,
        CenterConfig::Random(n_centers, center_box) => {
            if let Some(seed) = &random_state {
                let seed_rng = StdRng::from_seed(*seed);
                uniform_centers(n_centers, n_features, center_box, seed_rng)
            } else {
                uniform_centers(n_centers, n_features, center_box, OsRng)
            }
        }
    };
    let centers = centers_data.view();
    let samples_per_center = n_samples / centers.nrows();
    let mut data = vec![];
    if let Some(seed) = random_state {
        let seed_rng = StdRng::from_seed(seed);
        for center in centers.rows() {
            data.push(make_a_blob(
                center,
                samples_per_center,
                cluster_std,
                &seed_rng,
            ));
        }
    } else {
        for center in centers.rows() {
            data.push(make_a_blob(center, samples_per_center, cluster_std, &OsRng));
        }
    }
    let blobs: Vec<_> = data
        .iter()
        .map(|blob| {
            ArrayView::from_shape((n_features, samples_per_center), blob.as_slice())
                .expect("data generated incorrectly")
                .reversed_axes()
        })
        .collect();
    concatenate(Axis(0), blobs.as_slice()).expect("data generated incorrectly")
}

/// `make_a_blob`: generate an isotropic Gaussian blob,
///  centered at `center` with standard deviation `std_dev`
///  blob size: `n_smaples`
/// data is returned in form of Vec<f64> (COLUMN major: `n_features` * `n_samples`)
fn make_a_blob<R: RngCore + Clone>(
    center: ArrayView1<f64>,
    n_samples: usize,
    std_dev: f64,
    seed_rng: &R,
) -> Vec<f64> {
    let mut data = Vec::new();
    for c in center {
        let mut rng = StdRng::from_rng(seed_rng.clone()).unwrap();
        let norm = Normal::new(*c, std_dev).unwrap();
        data.extend(norm.sample_iter(&mut rng).take(n_samples));
    }
    data
}

/// `uniform_centers` Generate uniformly distributed `n_centers * n_features` centers
/// within the bounding box provided by `center_box`. Rng used is seeded by `seed_rng`.
/// results are returned in form of Vec<f64> (row major `n_centers` * `n_features`).
fn uniform_centers<R: RngCore>(
    n_centers: usize,
    n_features: usize,
    center_box: (f64, f64),
    seed_rng: R,
) -> Array2<f64> {
    let (low, high) = center_box;
    let mut rng = StdRng::from_rng(seed_rng).unwrap();
    let between = Uniform::new(low, high);
    let data = between
        .sample_iter(&mut rng)
        .take(n_centers * n_features)
        .collect();
    Array2::from_shape_vec((n_centers, n_features), data).unwrap()
}

mod test {
    use ndarray_rand::rand::rngs::OsRng;

    #[test]
    fn make_a_blob() {
        let center = ndarray::arr1(&[1., 1., 1.]);
        let n = 5;
        let blob = super::make_a_blob(center.view(), 5, 1., &OsRng);
        assert_eq!(blob.len(), center.ncols() * n);
    }

    #[test]
    fn uniform_centers() {
        let n = 5;
        let m = 3;
        let centers = super::uniform_centers(n, m, (-10., 10.), OsRng);
        assert_eq!(centers.nrows(), n);
        assert_eq!(centers.ncols(), m);
    }

    #[test]
    fn uniform_centers() {
        let n = 500;
        let dim = 3;

        let array = super::make_blobs(
            n,
            dim,
            None,
            None,
            Some(b"this is a test, this is a test. ".to_owned()),
        );
        assert_eq!(array.ncols(), 3);
        assert_eq!(array.nrows(), 500 / 3 * 3);
    }

    #[test]
    fn fixed_centers() {
        let n = 6;
        let dim = 3;
        let centers = ndarray::arr2(&[[1., 1., 1.], [-1., -1., -1.], [1., -1., 1.]]);
        let array = super::make_blobs(
            n,
            dim,
            Some(super::CenterConfig::Fixed(centers)),
            Some(0.4),
            Some(b"this is a test, this is a test. ".to_owned()),
        );
        assert_eq!(array.ncols(), 3);
        assert_eq!(array.nrows(), 6 / 3 * 3);
    }
}
