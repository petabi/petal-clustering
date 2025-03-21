mod dbscan;
mod hdbscan;
mod mst;
mod optics;
mod union_find;

pub use dbscan::Dbscan;
pub use hdbscan::HDbscan;
pub use optics::Optics;

/// An interface to train a model.
pub trait Fit<I, P, O>
where
    I: ?Sized,
{
    fn fit(&mut self, input: &I, params: Option<&P>) -> O;
}

/// An interface to apply a trained model.
pub trait Predict<I, O>
where
    I: ?Sized,
{
    fn predict(&mut self, input: &I) -> O;
}
