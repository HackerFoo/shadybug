pub mod derivative;
pub mod render;

use core::{
    fmt::Debug,
    future::Future,
    ops::{Add, Mul, Sub},
    task,
};

use glam::{UVec2, Vec2, Vec3};

use derivative::{Derivative, DerivativeCell};
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

impl<'a, T: Shader<FragmentInput: Clone>> Clone for Sampler<'a, T> {
    fn clone(&self) -> Self {
        Self {
            shader: self.shader,
            fragment_inputs: self.fragment_inputs.clone(),
            ndc: self.ndc.clone(),
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

impl<T: Shader> Sampler<'_, T> {
    /// Run the fragment shader for the given coordinates and offsets,
    /// running four threads in parallel to produce four results.
    pub fn get(
        &self,
        coord: Vec2,
        offsets: Vec2,
        inverse_offsets: Vec2,
    ) -> [Result<T::FragmentOutput, SamplerError<T::Error>>; 4] {
        let derivatives: [DerivativeCell<T::DerivativeType>; 4] = Default::default();
        let sample_offsets = [Vec2::ZERO, offsets.with_y(0.), offsets.with_x(0.), offsets];
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
        if barycentric.iter().all(|b| b.cmplt(Vec3::ZERO).any()) {
            discard4!();
        }
        // adjust barycentric weights for perspective
        //
        for b in &mut barycentric {
            *b /= self.inverse_z.dot(*b); // = *b * det / self.inverse_z.dot(*b * det)
        }
        let mut threads = [
            (barycentric[0], &derivatives[0]),
            (barycentric[1], &derivatives[1]),
            (barycentric[2], &derivatives[2]),
            (barycentric[3], &derivatives[3]),
        ]
        .map(|(barycentric, derivative)| {
            Box::pin(async move {
                let input = Interpolate3::interpolate3(&self.fragment_inputs, barycentric);
                let ndc = Interpolate3::interpolate3(&self.ndc, barycentric);
                self.shader
                    .fragment(input, ndc, barycentric, self.front_facing, |x| {
                        derivative.get_result(x)
                    })
                    .await
            })
        });

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
                if let Some(input) = src.0.get().get_input() {
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
            derivatives[0].0.set(Derivative::Output(dpdx[0], dpdy[0]));
            derivatives[1].0.set(Derivative::Output(dpdx[0], dpdy[1]));
            derivatives[2].0.set(Derivative::Output(dpdx[1], dpdy[0]));
            derivatives[3].0.set(Derivative::Output(dpdx[1], dpdy[1]));
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
pub trait Shader: Sized + Send + Sync {
    /// Input to vertex shader
    type Vertex;
    /// Output from vertex shader
    type VertexOutput;
    /// Input to fragment shader, interpolated from vertex output
    type FragmentInput: Interpolate3;
    /// Output from fragment shader
    type FragmentOutput;
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
        ndc: Vec2,
        barycentric: Vec3,
        front_facing: bool,
        derivative: F,
    ) -> impl Future<Output = Result<Self::FragmentOutput, SamplerError<Self::Error>>>
    where
        F: AsyncFn(
                Self::DerivativeType,
            ) -> Result<
                (Self::DerivativeType, Self::DerivativeType),
                SamplerError<Self::Error>,
            > + Copy;
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
#[derive(Debug)]
pub enum SamplerError<T = ()> {
    Discard,
    MissingSample,
    Other(T),
}
