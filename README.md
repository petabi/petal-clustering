# petal-clustering

A collection of clustering algorithms. Currently this crate provides DBSCAN, HDBSCAN and
OPTICS.

[![crates.io](https://img.shields.io/crates/v/petal-clustering)](https://crates.io/crates/petal-clustering)
[![Documentation](https://docs.rs/petal-clustering/badge.svg)](https://docs.rs/petal-clustering)
[![Coverage Status](https://codecov.io/gh/petabi/petal-clustering/branch/master/graphs/badge.svg)](https://codecov.io/gh/petabi/petal-clustering)

## Examples

The following example shows how to cluster points using DBSCAN.

```rust
use ndarray::array;
use petal_clustering::{Dbscan, Fit};

let points = array![[1.0, 2.0], [2.0, 2.0], [2.0, 2.3], [8.0, 7.0], [8.0, 8.0], [25.0, 80.0]];
let clustering = Dbscan::new(3.0, 2).fit(&points);

assert_eq!(clustering.0.len(), 2);        // two clusters found
assert_eq!(clustering.0[&0], [0, 1, 2]);  // the first three points in Cluster 0
assert_eq!(clustering.0[&1], [3, 4]);     // [8.0, 7.0] and [8.0, 8.0] in Cluster 1
assert_eq!(clustering.1, [5]);            // [25.0, 80.0] doesn't belong to any cluster
```

## Minimum Supported Rust Version

This crate is guaranteed to compile on Rust 1.53 and later.

## License

Copyright 2019-2022 Petabi, Inc.

Licensed under [Apache License, Version 2.0][apache-license] (the "License");
you may not use this crate except in compliance with the License.

Unless required by applicable law or agreed to in writing, software distributed
under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See [LICENSE](LICENSE) for
the specific language governing permissions and limitations under the License.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the [Apache-2.0
license][apache-license], shall be licensed as above, without any additional
terms or conditions.

[apache-license]: http://www.apache.org/licenses/LICENSE-2.0
