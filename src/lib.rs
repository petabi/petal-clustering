mod dbscan;

pub use dbscan::Dbscan;

pub trait Fit<'a> {
    type Input;
    type Output;

    fn fit(&mut self, input: Self::Input) -> Self::Output;
}

pub trait Predict<'a> {
    type Input;
    type Output;

    fn predict(&mut self, input: Self::Input) -> Self::Output;
}
