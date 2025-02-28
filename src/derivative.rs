use std::{
    cell::Cell,
    fmt::Debug,
    ops::Deref,
    pin::Pin,
    task::{Context, Poll},
};

use crate::SamplerError;

/// Inputs to and outputs from derivative calculation
#[derive(Clone, Copy, Debug)]
pub enum Derivative<T> {
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
pub struct DerivativeCell<T>(Cell<Derivative<T>>);

impl<T: Copy + Debug> Debug for DerivativeCell<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_tuple("DerivativeCell").field(&self.0).finish()
    }
}

unsafe impl<T> Send for DerivativeCell<T> {}
unsafe impl<T> Sync for DerivativeCell<T> {}

impl<T> Default for DerivativeCell<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T> Deref for DerivativeCell<T> {
    type Target = Cell<Derivative<T>>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> Derivative<T> {
    pub fn get_input(self) -> Option<T> {
        match self {
            Derivative::Input(input) => Some(input),
            _ => None,
        }
    }
    pub fn get_dpdx(self) -> Option<T> {
        match self {
            Derivative::Output(dpdx, _) => Some(dpdx),
            _ => None,
        }
    }
    pub fn get_dpdy(self) -> Option<T> {
        match self {
            Derivative::Output(_, dpdy) => Some(dpdy),
            _ => None,
        }
    }
}

impl<T: Copy + Debug + Default> DerivativeCell<T> {
    /// asynchronously compute the partial derivative with respect to X position
    pub async fn dpdx<E>(&self, value: T) -> Result<T, SamplerError<E>> {
        self.set(Derivative::Input(value));
        self.await.get_dpdx().ok_or(SamplerError::MissingSample)
    }
    /// asynchronously compute the partial derivative with respect to Y position
    pub async fn dpdy<E>(&self, value: T) -> Result<T, SamplerError<E>> {
        self.set(Derivative::Input(value));
        self.await.get_dpdy().ok_or(SamplerError::MissingSample)
    }
}

impl<'a, T: Copy + Default> Future for &'a DerivativeCell<T> {
    type Output = Derivative<T>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let val = self.get();
        match val {
            Derivative::Output(..) => Poll::Ready(val),
            _ => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
        }
    }
}
