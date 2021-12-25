mod dbscan;
mod optics;

use crate::dbscan::build as dbscan_build;
use crate::optics::build as optics_build;
use criterion::{criterion_group, criterion_main, Criterion};

criterion_group! {
name = benches;
config = Criterion::default()
    .sample_size(100)
    .measurement_time(std::time::Duration::new(60, 0));
targets = dbscan_build, optics_build}

criterion_main!(benches);
