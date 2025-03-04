pub mod basic;
pub mod channel;
pub mod render;

use core::{
    fmt::Debug,
    future::Future,
    ops::{Add, Mul, Sub},
    task,
};
use std::pin::pin;

use glam::{UVec2, Vec2, Vec3};

use channel::{BiChannel, InOut};
pub use render::{Bounds, SamplerTile, SamplerTileIter, ndc_to_pixel, render};

/// Discard a fragment
#[macro_export]
macro_rules! discard {
    () => {
        return Err(SamplerError::Discard);
    };
}

/// Used to sample four pixels at a time to get fragment output
pub struct Sampler<'a, T: Shader> {
    pub shader: &'a T,
    pub fragment_inputs: [T::FragmentInput; 3],
    pub ndc: [Vec2; 3],
    pub inverse_z: Vec3,
    pub front_facing: bool,
}

impl<T: Shader<FragmentInput: Clone>> Clone for Sampler<'_, T> {
    fn clone(&self) -> Self {
        Self {
            shader: self.shader,
            fragment_inputs: self.fragment_inputs.clone(),
            ndc: self.ndc,
            inverse_z: self.inverse_z,
            front_facing: self.front_facing,
        }
    }
}

/// Determinate of AB and AC, gives twice the area of the triangle ABC
fn det3(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b - a).perp_dot(c - a)
}

impl<'a, T: Shader> Sampler<'a, T> {
    /// Build a sampler from a shader and a triangle
    pub fn new(shader: &'a T, vertex_outputs: [T::VertexOutput; 3]) -> Self {
        let p = vertex_outputs.map(|v| T::perspective(v));
        let ndc = p.each_ref().map(|x| x.1);
        let inv_z = Vec3::from(p.each_ref().map(|x| x.2));
        let fragment_inputs = p.map(|x| x.0);
        let det = det3(ndc[0], ndc[1], ndc[2]);
        Self {
            shader,
            fragment_inputs,
            ndc,
            inverse_z: inv_z,
            front_facing: det >= 0.,
        }
    }
}

/// Define how to interpolate using barycentric coordinates (weights.)
pub trait Interpolate3 {
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self
    where
        Self: Sized;
}

impl<T> Interpolate3 for T
where
    T: Add<T, Output = T> + Mul<f32, Output = T> + Copy,
{
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self {
        input[0] * barycentric.x + input[1] * barycentric.y + input[2] * barycentric.z
    }
}

/// Discard all four threads
macro_rules! discard4 {
    () => {
        return [
            Err(SamplerError::Discard),
            Err(SamplerError::Discard),
            Err(SamplerError::Discard),
            Err(SamplerError::Discard),
        ];
    };
}

type DerivativeChannel<T> = BiChannel<T, (T, T)>;

impl<T: Shader> Sampler<'_, T> {
    /// Run the fragment shader for the given coordinates and offsets,
    /// running four threads in parallel to produce four results.
    pub fn get(
        &self,
        coord: Vec2,
        sample_offsets: &[Vec2; 4],
        inverse_offsets: Vec2,
    ) -> [Result<T::FragmentOutput, SamplerError<T::Error>>; 4] {
        let derivatives: [DerivativeChannel<T::DerivativeType>; 4] = Default::default();
        let mut barycentric = sample_offsets.map(|offset| {
            let coord = coord + offset;

            // barycentric coordinates, such that multiplying the weights by
            // the vertices will give the coordinates back in position.xy
            // NOTE: not scaled by det3 of self.ndc because it will be cancelled when adjusting for perspective
            Vec3::new(
                det3(coord, self.ndc[1], self.ndc[2]),
                det3(self.ndc[0], coord, self.ndc[2]),
                det3(self.ndc[0], self.ndc[1], coord),
            )
        });
        let outside = barycentric.each_ref().map(|b| b.cmplt(Vec3::ZERO).any());
        if outside.iter().all(|x| *x) {
            discard4!();
        }
        // adjust barycentric weights for perspective
        //
        for b in &mut barycentric {
            *b /= self.inverse_z.dot(*b); // = *b * det / self.inverse_z.dot(*b * det)
        }
        let [input0, input1, input2, input3] = barycentric
            .map(|barycentric| Interpolate3::interpolate3(&self.fragment_inputs, barycentric));
        let [barycentric0, barycentric1, barycentric2, barycentric3] = barycentric;
        let [thread0, thread1, thread2, thread3] = [
            (barycentric0, input0, &derivatives[0]),
            (barycentric1, input1, &derivatives[1]),
            (barycentric2, input2, &derivatives[2]),
            (barycentric3, input3, &derivatives[3]),
        ]
        .map(|(barycentric, input, derivative)| {
            self.shader
                .fragment(input, barycentric, self.front_facing, |x| {
                    derivative.write(x);
                    derivative.read()
                })
        });

        let mut threads = [pin!(thread0), pin!(thread1), pin!(thread2), pin!(thread3)];

        // run four threads and compute derivatives whenever they yield
        let mut results = [None, None, None, None];
        let mut context = task::Context::from_waker(task::Waker::noop());
        loop {
            let mut ready = true;
            for (thread, result) in threads.iter_mut().zip(results.iter_mut()) {
                if result.is_none() {
                    match Future::poll(thread.as_mut(), &mut context) {
                        task::Poll::Ready(r) => {
                            *result = Some(r);
                        }
                        task::Poll::Pending => {
                            ready = false;
                        }
                    }
                }
            }
            if ready {
                break;
            }

            let mut d: [T::DerivativeType; 4] = Default::default();
            for (src, dst) in derivatives.iter().zip(d.iter_mut()) {
                if let InOut::Input(input) = src.get() {
                    *dst = input;
                } else {
                    return results.map(|r| r.unwrap_or(Err(SamplerError::MissingSample)));
                }
            }

            // calculate derivatives
            let dpdx = [
                (d[1] - d[0]) * inverse_offsets.x,
                (d[3] - d[2]) * inverse_offsets.x,
            ];
            let dpdy = [
                (d[2] - d[0]) * inverse_offsets.y,
                (d[3] - d[1]) * inverse_offsets.y,
            ];
            derivatives[0].set(InOut::Output((dpdx[0], dpdy[0])));
            derivatives[1].set(InOut::Output((dpdx[0], dpdy[1])));
            derivatives[2].set(InOut::Output((dpdx[1], dpdy[0])));
            derivatives[3].set(InOut::Output((dpdx[1], dpdy[1])));
        }
        for (result, outside) in results.iter_mut().zip(outside.into_iter()) {
            if outside {
                *result = Some(Err(SamplerError::Discard));
            }
        }
        results.map(Option::unwrap)
    }
    /// Calculate the bounds in pixels
    pub fn bounds(&self, pixels: u32) -> Bounds<UVec2> {
        if !self.front_facing {
            Bounds::Zero
        } else {
            Bounds::Bounds {
                lo: ndc_to_pixel(
                    self.ndc.iter().fold(-Vec2::ONE, |acc, v| acc.min(*v)),
                    pixels,
                ),
                hi: ndc_to_pixel(
                    self.ndc.iter().fold(Vec2::ONE, |acc, v| acc.max(*v)),
                    pixels,
                ),
            }
        }
    }
}

/// Define vertex and fragment shaders
pub trait Shader: Sized {
    /// Input to vertex shader
    type Vertex;
    /// Output from vertex shader
    type VertexOutput;
    /// Input to fragment shader, interpolated from vertex output
    type FragmentInput: Interpolate3;
    /// Output from fragment shader
    type FragmentOutput;
    /// Per sample data
    type Sample: Default + Clone;
    /// Target for data to be finally output
    type Target;
    /// User errors from fragment shader
    type Error;
    /// Type of derivatives
    /// Use an enum if multiple types are needed.
    type DerivativeType: Default
        + Copy
        + Sub<Self::DerivativeType, Output = Self::DerivativeType>
        + Mul<f32, Output = Self::DerivativeType>;
    /// Vertex shader
    fn vertex(&self, vertex: &Self::Vertex) -> Self::VertexOutput;
    /// Perspective division and conversion of vertex outputs to fragment inputs for interpolation
    /// Returns a fragment input, normalized device coordinates, and the inverse depth
    fn perspective(vertex_output: Self::VertexOutput) -> (Self::FragmentInput, Vec2, f32);
    /// Fragment shader
    fn fragment<F>(
        &self,
        input: Self::FragmentInput,
        barycentric: Vec3,
        front_facing: bool,
        derivative: F,
    ) -> impl Future<Output = Result<Self::FragmentOutput, SamplerError<Self::Error>>>
    where
        F: AsyncFn(Self::DerivativeType) -> (Self::DerivativeType, Self::DerivativeType) + Copy;
    /// Combine fragment data into a pixel
    /// Performs depth test and blending
    fn combine(fragment: Self::FragmentOutput, sample: &mut Self::Sample);
    /// Consume an iterator over the tile with the given offset and size and write to the target
    fn merge<I: Iterator<Item = (UVec2, Self::Sample)>>(
        offset: UVec2,
        size: UVec2,
        iter: I,
        target: &mut Self::Target,
    );
    /// Create a sampler to draw the given triangle defined by three vertex indices
    fn draw_triangle<'a>(
        &'a self,
        vertices: &'a [Self::Vertex],
        indices: &'a [usize],
    ) -> Sampler<'a, Self>
    where
        Self: Sized,
    {
        let vertex_outputs = [
            self.vertex(&vertices[indices[0]]),
            self.vertex(&vertices[indices[1]]),
            self.vertex(&vertices[indices[2]]),
        ];
        Sampler::new(self, vertex_outputs)
    }
    /// Iterate over tiles of size at most `max_size` pixels square
    fn tiled_iter<'a>(
        &'a self,
        vertices: &'a [Self::Vertex],
        indices: &'a [usize],
        pixels: u32,
        max_size: u32,
    ) -> SamplerTileIter<'a, Self> {
        let samplers: Vec<_> = indices
            .chunks(3)
            .map(|indices| self.draw_triangle(vertices, indices))
            .collect();
        SamplerTileIter {
            max_size,
            tiles: vec![SamplerTile {
                offset: UVec2::ZERO,
                size: UVec2::splat(pixels),
                pixels,
                samplers,
            }],
        }
    }
}

/// Stage of fragment shader, used to compute derivatives
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Stage {
    Normal,
    SampleX,
    SampleY,
    UseSamples,
}

/// Invalid fragment shader outputs
#[derive(Debug, Clone, Copy)]
pub enum SamplerError<T = ()> {
    Discard,
    MissingSample,
    Other(T),
}
