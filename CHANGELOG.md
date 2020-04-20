# Changelog

This file documents recent notable changes to this project. The format of this
file is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and
this project adheres to [Semantic
Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/petabi/petal-clustering/compare/0.2.1...master
[0.2.1]: https://github.com/petabi/petal-clustering/compare/0.2.0...0.2.1
[0.2.0]: https://github.com/petabi/petal-clustering/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/petabi/petal-clustering/tree/0.1.0
