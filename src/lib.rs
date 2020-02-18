mod dbscan;
mod optics;

pub use dbscan::Dbscan;
pub use optics::Optics;

/// An interface to train a model.
pub trait Fit<'a> {
    type Input;
    type Output;

    fn fit(&mut self, input: Self::Input) -> Self::Output;
}

/// An interface to apply a trained model.
pub trait Predict<'a> {
    type Input;
    type Output;

    fn predict(&mut self, input: Self::Input) -> Self::Output;
}
