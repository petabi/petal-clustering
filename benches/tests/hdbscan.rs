use super::setup::{make_blobs, CenterConfig};
use criterion::{black_box, Criterion};
use ndarray::{arr2, ArrayView};
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_clustering::{Fit, HDbscan};

pub fn build(c: &mut Criterion) {
    let n = black_box(5000);
    let dim = black_box(3);

    let mut rng = StdRng::from_seed(*b"ball tree build bench test seed ");
    let data: Vec<f64> = (0..n * dim).map(|_| rng.gen()).collect();
    let array = ArrayView::from_shape((n, dim), &data).unwrap();
    c.bench_function("hdbscan::build", |b| {
        b.iter(|| {
            let mut model = HDbscan::default();
            model.fit(&array);
        })
    });
}

pub fn uniform_clusters(c: &mut Criterion) {
    let n = black_box(500);
    let dim = black_box(3);

    let array = make_blobs(n, dim, None, None, None);
    c.bench_function("hdbscan::uniform_clusters", |b| {
        b.iter(|| {
            let mut model = HDbscan::default();
            model.fit(&array.view());
        })
    });
}

pub fn fixed_clusters(c: &mut Criterion) {
    let n = black_box(500);
    let dim = black_box(3);
    let centers = arr2(&[[1., 1., 1.], [-1., -1., -1.], [1., -1., 1.]]);

    let array = make_blobs(n, dim, Some(CenterConfig::Fixed(centers)), Some(0.4), None);

    c.bench_function("hdbscan::fixed_clusters", |b| {
        b.iter(|| {
            let mut model = HDbscan::default();
            model.fit(&array.view());
        })
    });
}
