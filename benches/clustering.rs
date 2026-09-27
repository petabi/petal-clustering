//! End-to-end clustering benchmarks for tracking performance changes.

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::Array2;
use ndarray_rand::rand::{rngs::StdRng, Rng, SeedableRng};
use petal_clustering::{Dbscan, Fit, HDbscan, Optics};
use petal_neighbors::distance::Euclidean;

const POINTS: usize = 256;
const DIMENSIONS: [usize; 3] = [16, 64, 256];

macro_rules! benchmark_type {
    ($name:ident, $type:ty, $label:literal) => {
        fn $name(c: &mut Criterion) {
            // Keep this seed unchanged so existing Criterion baselines remain comparable.
            let mut rng = StdRng::from_seed(*b"clustering SIMD benchmark seed!!");

            for dim in DIMENSIONS {
                // Four separated, uniformly distributed groups. Generate the
                // data once so every algorithm and dependency version sees
                // the same contiguous rows and neighborhood density.
                let input = Array2::from_shape_fn((POINTS, dim), |(row, _)| {
                    ((row % 4) as f64 * 3.0 + rng.random::<f64>()) as $type
                });
                let eps = ((dim as f64 / 6.0).sqrt() * 1.1) as $type;

                let mut group = c.benchmark_group(format!("clustering/{label}", label = $label));
                group.throughput(Throughput::Elements(POINTS as u64));

                group.bench_with_input(BenchmarkId::new("dbscan", dim), &input, |b, input| {
                    b.iter(|| {
                        let mut model = Dbscan::new(eps, 8, Euclidean::default());
                        black_box(model.fit(black_box(input), None))
                    });
                });

                group.bench_with_input(BenchmarkId::new("optics", dim), &input, |b, input| {
                    b.iter(|| {
                        let mut model = Optics::new(eps, 8, Euclidean::default());
                        black_box(model.fit(black_box(input), None))
                    });
                });

                for (name, boruvka) in [("hdbscan_boruvka", true), ("hdbscan_prim", false)] {
                    group.bench_with_input(BenchmarkId::new(name, dim), &input, |b, input| {
                        b.iter(|| {
                            let mut model = HDbscan {
                                alpha: 1.0,
                                min_samples: 8,
                                min_cluster_size: 8,
                                metric: Euclidean::default(),
                                boruvka,
                            };
                            black_box(model.fit(black_box(input), None))
                        });
                    });
                }
                group.finish();
            }
        }
    };
}

benchmark_type!(f32_clustering, f32, "f32");
benchmark_type!(f64_clustering, f64, "f64");

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = f32_clustering, f64_clustering
}
criterion_main!(benches);
