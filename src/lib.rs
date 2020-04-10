mod dbscan;
mod optics;

pub use dbscan::Dbscan;
pub use optics::Optics;

/// An interface to train a model.
pub trait Fit<I, O> {
    fn fit(&mut self, input: &I) -> O;
}

/// An interface to apply a trained model.
pub trait Predict<I, O> {
    fn predict(&mut self, input: &I) -> O;
}
