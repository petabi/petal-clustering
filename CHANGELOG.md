# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - 2024-02-21

### Changed

- Requires Rust 1.74 or later.

### Fixed

- Corrected the dual-tree traversal order in the `HDbscan` implementation when
  `boruvka: true` is set. Previously, this misplacement caused `HDbscan` with
  `boruvka: true` to produce a heavier Minimum Spanning Tree (MST) than when
  `boruvka: false` was set. (PR #67 from @azizkayumov)

## [0.7.0] - 2023-12-21

### Changed

- Requires Rust 1.70 or later.

## [0.6.0] - 2023-08-07

### Changed

- Switched to [Rust 2021 Edition](https://doc.rust-lang.org/edition-guide/rust-2021).
- Requires Rust 1.64 or later.

### Fixed

- Cluster that is smaller than `min_samples` for `Dbscan` should become outliers.

## [0.5.1] - 2022-05-03

### Changed

- `HDbscan` now allows non-standard layout matrix as input.

## [0.5.0] - 2022-04-20

### Added

- The [HDBSCAN] clustering algorithm.

### Changed

- Update MSRV to 1.53.0.

## [0.4.0] - 2021-07-07

### Changed

- `Dbscan` and `Optics` now allows caller to designate customized `Metric`
  to compute distance, `Default` is using `Euclidean`.

## [0.3.0] - 2020-03-29

### Changed

- Requires Rust 1.49 or later.
- Upgrade ndarray to 0.15.0

## [0.2.3] - 2020-12-31

### Changed

- Upgrade ndarray to 0.14.0
- Upgrade petal-neighbors to 0.5.1

## [0.2.2] - 2020-08-28

### Added

- `Dbscan` and `Optics` can accept an array of `f32`.

### Documentation

- Examples added to `Dbscan` and `Optics`.

## [0.2.1] - 2020-04-13

### Changed

- The input type of `Fit` and `Predict` no longer has to be `Sized`. With this
  chagne, the caller may pass a slice as an input.

## [0.2.0] - 2020-04-10

### Changed

- Clustering algorithms takes `ArrayBase` as its input, instead of `ArrayView`,
  to allow more types in ndarray.

## [0.1.0] - 2020-02-18

### Added

- The [DBSCAN](https://en.wikipedia.org/wiki/DBSCAN) clustering algorithm.
- The [OPTICS](https://en.wikipedia.org/wiki/OPTICS_algorithm) clustering
  algorithm.

[0.8.0]: https://github.com/petabi/petal-clustering/compare/0.7.0...0.8.0
[0.7.0]: https://github.com/petabi/petal-clustering/compare/0.6.0...0.7.0
[0.6.0]: https://github.com/petabi/petal-clustering/compare/0.5.1...0.6.0
[0.5.1]: https://github.com/petabi/petal-clustering/compare/0.5.0...0.5.1
[0.5.0]: https://github.com/petabi/petal-clustering/compare/0.4.0...0.5.0
[0.4.0]: https://github.com/petabi/petal-clustering/compare/0.3.0...0.4.0
[0.3.0]: https://github.com/petabi/petal-clustering/compare/0.2.3...0.3.0
[0.2.2]: https://github.com/petabi/petal-clustering/compare/0.2.2...0.2.3
[0.2.1]: https://github.com/petabi/petal-clustering/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/petabi/petal-clustering/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/petabi/petal-clustering/tree/0.1.0

[HDBSCAN]: https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html
