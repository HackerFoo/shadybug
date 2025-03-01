pub mod derivative;
pub mod render;

use core::{
    fmt::Debug,
    future::Future,
    ops::{Add, Div, Mul, Sub}, task,
};

use glam::{UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};

use derivative::{Derivative, DerivativeCell};
pub use render::{ndc_to_pixel, Bounds, SamplerTile, SamplerTileIter, HasColor, HasDepth, render};

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
    pub det: f32,
}

impl<'a, T: Shader<FragmentInput: Clone>> Clone for Sampler<'a, T> {
    fn clone(&self) -> Self {
        Self {
            shader: self.shader,
            fragment_inputs: self.fragment_inputs.clone(),
            ndc: self.ndc.clone(),
            inverse_z: self.inverse_z,
            det: self.det,
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
            det,
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
    ) -> [Result<T::FragmentOutput, SamplerError<T::Error>>; 4] {
        let derivatives: [DerivativeCell<T::DerivativeType>; 4] = Default::default();
        let sample_offsets = [Vec2::ZERO, offsets.with_y(0.), offsets.with_x(0.), offsets];
        let mut barycentric = sample_offsets.map(|offset| {
            let coord = coord + offset;

            // barycentric coordinates, such that multiplying the weights by
            // the vertices will give the coordinates back in position.xy
            Vec3::new(
                det3(coord, self.ndc[1], self.ndc[2]),
                det3(self.ndc[0], coord, self.ndc[2]),
                det3(self.ndc[0], self.ndc[1], coord),
            ) / self.det
        });
        if barycentric.iter().all(|b| b.cmplt(Vec3::ZERO).any()) {
            discard4!();
        }
        // adjust barycentric weights for perspective
        for b in &mut barycentric {
            *b /= self.inverse_z.dot(*b);
        }
        let mut context = task::Context::from_waker(task::Waker::noop());
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
                    .fragment(input, ndc, barycentric, self.det >= 0., |x| derivative.get_result(x))
                    .await
            })
        });

        // run four threads and compute derivatives whenever they yield
        let mut results = [None, None, None, None];
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
            let dpdx = [(d[1] - d[0]) / offsets.x, (d[3] - d[2]) / offsets.x];
            let dpdy = [(d[2] - d[0]) / offsets.y, (d[3] - d[1]) / offsets.y];
            derivatives[0].0.set(Derivative::Output(dpdx[0], dpdy[0]));
            derivatives[1].0.set(Derivative::Output(dpdx[0], dpdy[1]));
            derivatives[2].0.set(Derivative::Output(dpdx[1], dpdy[0]));
            derivatives[3].0.set(Derivative::Output(dpdx[1], dpdy[1]));
        }
        results.map(Option::unwrap)
    }
    /// Calculate the bounds in pixels
    pub fn bounds(&self, pixels: u32) -> Bounds<UVec2> {
        if self.det < 0. {
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
    type Vertex;
    type VertexOutput;
    type FragmentInput: Interpolate3;
    type FragmentOutput;
    type Error;
    type DerivativeType: Default
        + Debug
        + Copy
        + Sub<Self::DerivativeType, Output = Self::DerivativeType>
        + Div<f32, Output = Self::DerivativeType>;
    fn vertex(&self, vertex: &Self::Vertex) -> Self::VertexOutput;
    fn perspective(vertex_output: Self::VertexOutput) -> (Self::FragmentInput, Vec2, f32);
    fn fragment<F>(
        &self,
        input: Self::FragmentInput,
        ndc: Vec2,
        barycentric: Vec3,
        front_facing: bool,
        derivative: F,
    ) -> impl Future<Output = Result<Self::FragmentOutput, SamplerError<Self::Error>>>
    where F: AsyncFn(Self::DerivativeType) -> Result<(Self::DerivativeType, Self::DerivativeType), SamplerError<Self::Error>> + Copy;
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
