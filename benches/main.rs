mod tests;

use criterion::{criterion_group, criterion_main, Criterion};
use tests::{
    dbscan_build, dbscan_fixed_clusters, dbscan_uniform_clusters, optics_build,
    optics_fixed_clusters, optics_uniform_clusters,
};

criterion_group! {
name = benches;
config = Criterion::default()
    .sample_size(100)
    .measurement_time(std::time::Duration::new(60, 0));
targets =
    dbscan_build, dbscan_fixed_clusters, dbscan_uniform_clusters,
    optics_build, optics_fixed_clusters, optics_uniform_clusters
}

criterion_main!(benches);
