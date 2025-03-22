mod dbscan;
mod hdbscan;
mod mst;
mod optics;
mod union_find;

pub use dbscan::Dbscan;
pub use hdbscan::HDbscan;
pub use optics::Optics;

/// An interface to train a model.
///
/// # Parameters
/// - `input`: A reference to the input data of type `I` that the model will fit.
/// - `params`: An optional reference to parameters of type `P` that can be used to configure the fitting process.
///
/// # Returns
/// - Returns an output of type `O` which represents the result of the fitting process.
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
