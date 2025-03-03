use core::{
    cell::Cell,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

#[derive(Clone, Copy)]
pub(crate) enum InOut<In, Out> {
    Input(In),
    Output(Out),
    Empty,
}

impl<In, Out> Default for InOut<In, Out> {
    fn default() -> Self {
        Self::Empty
    }
}

/// A channel that can be read and written
pub struct BiChannel<In, Out>(pub(crate) Cell<InOut<In, Out>>);

impl<In, Out> Default for BiChannel<In, Out> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<In: Copy, Out: Copy> BiChannel<In, Out> {
    pub fn write(&self, input: In) {
        self.0.set(InOut::Input(input));
    }
    pub async fn read(&self) -> Out {
        self.await
    }
}

impl<'a, In: Copy, Out: Copy> Future for &'a BiChannel<In, Out> {
    type Output = Out;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.0.get() {
            InOut::Output(out) => Poll::Ready(out),
            _ => Poll::Pending,
        }
    }
}
