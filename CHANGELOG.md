# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Updated ndarray to 0.17 and petal-neighbors to 0.18.

### Fixed

- Fixed merge inconsistency in `HDbscan` where the labeling process did not
  consider ties (edges with equal weights). This could result in inconsistent
  cluster merges while building the hierarchy, with some points being connected
  to child clusters instead of their true parent clusters. The fix breaks ties
  using subtree sizes, ensuring clusters are merged first before points are
  merged. This particularly improves consistency with high dimensionality and
  higher values of `min_samples` where equally weighted distances are common.
  (PR #94 from @azizkayumov)

## [0.12.0] - 2025-05-03

### Added

- Implemented semi-supervised clustering capability for the `HDbscan` algorithm.
  This allows users to provide partially labeled data (as an
  `Option<&HashMap<usize, Vec<usize>>>`) to the `fit` method. The algorithm
  leverages these known labels using the BCubed metric to guide cluster
  formation, aiming to place points with the same label into the same cluster
  while still discovering density-based clusters in the unlabeled data. Test
  results on the Digits dataset show improved Adjusted Rand Index (ARI) scores
  compared to unsupervised HDBSCAN when using 10% partial labels, aligning
  closely with the reference Python implementation.

### Changed

- Requires Rust 1.81 or later.
- Updated petal-neighbors to 0.13.0.
- The `Fit` trait signature has been modified to support optional parameters
  during fitting. The `fit` method now accepts an additional `params:
  Option<&P>` argument:
  - Old signature: `fn fit(&mut self, input: &I) -> O;`
  - New signature: `fn fit(&mut self, input: &I, params: Option<&P>) -> O;`
- Calls to `fit` for `HDbscan`, `Dbscan`, and `Optics` need to be updated.
  - For `HDbscan`, pass `Some(&partial_labels_map)` to enable semi-supervised
    mode or `None` for standard unsupervised clustering.
  - For `Dbscan` and `Optics`, the `params` argument is currently unused; pass `None`.

## [0.11.0] - 2025-03-05

### Changed

- `HDbscan::fit` now returns an additional `Vec` that contains the score of
  each outlier. The score is computed from the [GLOSH] (Global-Local Outlier
  Score from Hierarchies) algorithm.
- `HDbscan::fit` now gracefully handles empty inputs and single-point inputs,
  returning empty clusters for these edge cases.

[GLOSH]: https://dl.acm.org/doi/10.1145/2733381

### Removed

- The dummy parameter HDbscan::eps has been removed as it was *NOT* being used
  in the current implementation, and the removal will *NOT* affect the
  clustering result. The current cluster selection method follows the "eom"
  (excess of mass) approach. This assumes an equivalent setting to
  cluster_selection_eps=0.0 and cluster_selection_method="eom" in
  [Scikit-learn]'s HDBSCAN implementation. Removing this unused parameter helps
  clarify the clustering behavior and avoids confusion.

[Scikit-learn]: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.HDBSCAN.html#sklearn.cluster.HDBSCAN

## [0.10.0] - 2024-11-01

### Changed

- `HDbscan` now accepts `f32`, in addion to `f64`, as the element type of the
  input matrix.

## [0.9.0] - 2024-08-09

### Changed

- Updated ndarray to 0.16.0.
- Updated petal-neighbors to 0.11.0

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

[Unreleased]: https://github.com/petabi/petal-clustering/compare/0.12.0...main
[0.12.0]: https://github.com/petabi/petal-clustering/compare/0.11.0...0.12.0
[0.11.0]: https://github.com/petabi/petal-clustering/compare/0.10.0...0.11.0
[0.10.0]: https://github.com/petabi/petal-clustering/compare/0.9.0...0.10.0
[0.9.0]: https://github.com/petabi/petal-clustering/compare/0.8.0...0.9.0
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
