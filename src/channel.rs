use core::{
    cell::Cell,
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};
use std::ops::Deref;

#[derive(Clone, Copy)]
pub enum InOut<In, Out> {
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
pub struct BiChannel<In, Out>(pub Cell<InOut<In, Out>>);

pub type Consumer<In> = BiChannel<In, ()>;
pub type Producer<Out> = BiChannel<(), Out>;

impl<In, Out> Deref for BiChannel<In, Out> {
    type Target = Cell<InOut<In, Out>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<In, Out> Default for BiChannel<In, Out> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<In: Copy, Out: Copy> BiChannel<In, Out> {
    pub fn write(&self, input: In) {
        self.set(InOut::Input(input));
    }
    pub async fn read(&self) -> Out {
        self.await
    }
}

impl<In: Copy, Out: Copy> Future for &BiChannel<In, Out> {
    type Output = Out;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.get() {
            InOut::Output(out) => {
                self.0.set(InOut::Empty);
                Poll::Ready(out)
            }
            _ => Poll::Pending,
        }
    }
}
