mod dbscan;
mod optics;

pub use dbscan::Dbscan;
pub use optics::Optics;

/// An interface to train a model.
pub trait Fit<I, O>
where
    I: ?Sized,
{
    fn fit(&mut self, input: &I) -> O;
}

/// An interface to apply a trained model.
pub trait Predict<I, O>
where
    I: ?Sized,
{
    fn predict(&mut self, input: &I) -> O;
}
