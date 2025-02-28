use core::{
    future::Future,
    cell::Cell,
    fmt::Debug,
    pin::Pin,
    task::{Context, Poll},
};

use crate::SamplerError;

/// Inputs to and outputs from derivative calculation
#[derive(Clone, Copy, Debug)]
pub(crate) enum Derivative<T> {
    Input(T),
    Output(T, T),
    Invalid,
}

impl<T> Default for Derivative<T> {
    fn default() -> Self {
        Self::Invalid
    }
}

/// A cell that can be updated with derivative data
pub(crate) struct DerivativeCell<T>(pub(crate) Cell<Derivative<T>>);

impl<T: Copy + Debug> Debug for DerivativeCell<T> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_tuple("DerivativeCell").field(&self.0).finish()
    }
}

impl<T> Default for DerivativeCell<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Derivative<T> {
    pub(crate) fn get_input(self) -> Option<T> {
        match self {
            Derivative::Input(input) => Some(input),
            _ => None,
        }
    }
    pub(crate) fn get_output(self) -> Option<(T, T)> {
        match self {
            Derivative::Output(dpdx, dpdy) => Some((dpdx, dpdy)),
            _ => None,
        }
    }
}

impl<T: Copy + Debug + Default> DerivativeCell<T> {
    /// asynchronously compute the partial derivatives with respect to position
    pub(crate) async fn get_result<E>(&self, value: T) -> Result<(T, T), SamplerError<E>> {
        self.0.set(Derivative::Input(value));
        self.await.get_output().ok_or(SamplerError::MissingSample)
    }
}

impl<'a, T: Copy + Default> Future for &'a DerivativeCell<T> {
    type Output = Derivative<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let val = self.0.get();
        match val {
            Derivative::Output(..) => Poll::Ready(val),
            _ => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
