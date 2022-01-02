use criterion::{black_box, Criterion};
use ndarray::{arr2, ArrayView};
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_clustering::{make_blobs, CenterConfig, Dbscan, Fit};
use petal_neighbors::distance::Euclidean;

#[allow(dead_code)]
pub(crate) fn build(c: &mut Criterion) {
    let n = black_box(5000);
    let dim = black_box(3);

    let mut rng = StdRng::from_seed(*b"ball tree build bench test seed ");
    let data: Vec<f64> = (0..n * dim).map(|_| rng.gen()).collect();
    let array = ArrayView::from_shape((n, dim), &data).unwrap();
    c.bench_function("build", |b| {
        b.iter(|| {
            let mut model = Dbscan::new(0.5, 10, Euclidean::default());
            model.fit(&array);
        })
    });
}

#[allow(dead_code)]
pub(crate) fn uniform_clusters(c: &mut Criterion) {
    let n = black_box(5000);
    let dim = black_box(3);

    let array = make_blobs(n, dim, None, None);
    c.bench_function("uniform_clusters", |b| {
        b.iter(|| {
            let mut model = Dbscan::new(1., 10, Euclidean::default());
            model.fit(&array.view());
        })
    });
}

#[allow(dead_code)]
pub(crate) fn fixed_clusters(c: &mut Criterion) {
    let n = black_box(50_000);
    let dim = black_box(3);
    let centers = arr2(&[[1., 1., 1.], [-1., -1., -1.], [1., -1., 1.]]);

    let array = make_blobs(n, dim, Some(CenterConfig::Fixed(centers)), Some(0.4));

    c.bench_function("fixed_clusters", |b| {
        b.iter(|| {
            let mut model = Dbscan::new(0.3, 10, Euclidean::default());
            model.fit(&array.view());
        })
    });
}
