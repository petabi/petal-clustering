mod dbscan;

pub use dbscan::DBSCAN;

pub trait Fit {
    type Input;
    type Output;

    fn fit(&mut self, input: Self::Input) -> Self::Output;
}

pub trait Predict {
    type Input;
    type Output;

    fn predict(&mut self, input: Self::Input) -> Self::Output;
}
