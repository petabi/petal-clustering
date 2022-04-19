use csv::ReaderBuilder;
use ndarray::Array2;
use petal_clustering::{Fit, HDbscan};
use petal_neighbors::distance::Euclidean;
use std::{env, fs::File, process::exit};

fn main() {
    let (file, min_cluster_size, min_samples) = parse();
    let data_file = File::open(file).expect("file open failed");
    let mut rdr = ReaderBuilder::new()
        .has_headers(false)
        .from_reader(data_file);
    let mut nfeatures = 0;
    let data: Vec<f64> = rdr
        .deserialize()
        .map(|v| {
            let r: Vec<f64> = v.expect("corruptted data");
            if nfeatures < 1 {
                nfeatures = r.len();
            }
            r.into_iter()
        })
        .flatten()
        .collect();
    if nfeatures < 1 {
        println!(
            "data file is too small: {} feature(s) detected, {} entries in total",
            nfeatures,
            data.len()
        );
        exit(0);
    }
    let nevents = data.len() / nfeatures;
    let data = Array2::from_shape_vec((nevents, nfeatures), data).expect("data shape error");
    let mut clustering = HDbscan {
        eps: 0.5,
        alpha: 1.,
        min_samples,
        min_cluster_size,
        metric: Euclidean::default(),
        boruvka: true,
    };
    let (clusters, outliers) = clustering.fit(&data.view());
    println!("========= Report =========");
    println!("# of events processed: {}", data.nrows());
    println!("# of features provided: {}", data.ncols());
    println!("# of clusters: {}", clusters.len());
    println!(
        "# of events clustered: {}",
        clusters.values().map(|v| v.len()).sum::<usize>(),
    );
    println!("# of outliers: {}", outliers.len());
}

fn parse() -> (String, usize, usize) {
    let args = env::args().collect::<Vec<_>>();

    if args.len() <= 1 || args[1] == "--help" || args[1] == "-h" {
        help();
        exit(0);
    }

    let path = args.last().expect("unable to detect data file").clone();
    if args.len() < 3 {
        return (path, 15, 15);
    }

    if args.len() == 5 && (args[1] == "--params" || args[1] == "-p") {
        let min_cluster_size: usize = args[2].parse().unwrap_or(15);
        let min_samples: usize = args[3].parse().unwrap_or(15);
        return (path, min_cluster_size, min_samples);
    }

    println!("unable to process provided arguments: ");
    for (nth, arg) in args.iter().enumerate().take(4) {
        println!("{}. {:?}", nth, arg);
    }
    exit(0);
}

fn help() {
    println!(
        "USAGE: \
        \n hdbscan [DATAFILE] \
        \n \
        \nFlags: \
        \n    -h, --help       Prints help information \
        \n \
        \nARG: \
        \n    -p, --params <min_cluster_size> <min_samples> \
        \n                  Sets min_cluster_size and min_samples \
        \n    <DATAFILE>    A CSV data file that satisfies the following: \
        \n                  1) No header line \
        \n                  2) `,` as delimiter \
        \n                  3) data can be accepted by `str::parse::<f64>` \
        \n                     only (e.g. `%.2f`)"
    );
}
