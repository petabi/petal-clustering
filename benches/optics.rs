use criterion::{black_box, Criterion};
use ndarray::ArrayView;
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_clustering::{Fit, Optics};
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
            let mut model = Optics::new(0.5, 10, Euclidean::default());
            model.fit(&array);
        })
    });
}
