#![feature(portable_simd)]

pub mod derivative;
pub mod render;

use core::{
    fmt::Debug,
    future::Future,
    ops::{Add, Div, Mul, Sub}, task,
    simd::f32x4,
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

/// Used to sample one pixel at a time to get fragment output
pub struct Sampler<'a, T: Shader> {
    pub shader: &'a T,
    pub vertex_outputs: [T::VertexOutput; 3],
    pub ndc: [Vec4; 3],
    pub det: f32,
}

impl<'a, T: Shader> Clone for Sampler<'a, T> {
    fn clone(&self) -> Self {
        Self {
            shader: self.shader,
            vertex_outputs: self.vertex_outputs.clone(),
            ndc: self.ndc.clone(),
            det: self.det,
        }
    }
}

/// Determinate of AB and AC, gives twice the area of the triangle ABC
fn det3(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b - a).perp_dot(c - a)
}

#[derive(Debug, Clone, Copy)]
struct WideVec2([f32x4; 2]);

impl From<[Vec2; 4]> for WideVec2 {
    fn from(a: [Vec2; 4]) -> Self {
        WideVec2([
            f32x4::from_array([a[0].x, a[1].x, a[2].x, a[3].x]),
            f32x4::from_array([a[0].y, a[1].y, a[2].y, a[3].y])
        ])
    }
}

impl From<Vec2> for WideVec2 {
    fn from(a: Vec2) -> Self {
        WideVec2([
            f32x4::splat(a.x),
            f32x4::splat(a.y),
        ])
    }
}

impl Into<[Vec2; 4]> for WideVec2 {
    fn into(self) -> [Vec2; 4] {
        let [x, y] = self.0;
        let [ax, bx, cx, dx] = x.to_array();
        let [ay, by, cy, dy] = y.to_array();
        [
            Vec2::new(ax, ay),
            Vec2::new(bx, by),
            Vec2::new(cx, cy),
            Vec2::new(dx, dy),
        ]
    }
}

impl Add<Self> for WideVec2 {
    type Output = Self;
    fn add(self, WideVec2([b_x, b_y]): Self) -> Self::Output {
        let [a_x, a_y] = self.0;
        WideVec2([
            a_x + b_x,
            a_y + b_y
        ])
    }
}

impl Sub<Self> for WideVec2 {
    type Output = Self;
    fn sub(self, WideVec2([b_x, b_y]): Self) -> Self::Output {
        let [a_x, a_y] = self.0;
        WideVec2([
            a_x - b_x,
            a_y - b_y
        ])
    }
}

impl WideVec2 {
    pub fn perp_dot(self, WideVec2([b_x, b_y]): Self) -> f32x4 {
        let [a_x, a_y] = self.0;
        a_x * b_y - a_y * b_x
    }
}

fn wide_det3(a: WideVec2, b: WideVec2, c: WideVec2) -> f32x4 {
    (b - a).perp_dot(c - a)
}

impl<'a, T: Shader> Sampler<'a, T> {
    /// Build a sampler from a shader and a triangle
    pub fn new(shader: &'a T, vertex_outputs: [T::VertexOutput; 3]) -> Self {
        let ndc = [
            vertex_outputs[0].ndc(),
            vertex_outputs[1].ndc(),
            vertex_outputs[2].ndc(),
        ];
        let det = det3(ndc[0].xy(), ndc[1].xy(), ndc[2].xy());
        Self {
            shader,
            vertex_outputs,
            ndc,
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
        let coords4 = WideVec2::from([Vec2::ZERO, offsets.with_y(0.), offsets.with_x(0.), offsets]) + WideVec2::from(coord);
        /*
        let barycentric = sample_offsets.map(|offset| {
            let coord = coord + offset;

            // barycentric coordinates, such that multiplying the weights by
            // the vertices will give the coordinates back in position.xy
            Vec3::new(
                det3(coord, self.ndc[1].xy(), self.ndc[2].xy()),
                det3(self.ndc[0].xy(), coord, self.ndc[2].xy()),
                det3(self.ndc[0].xy(), self.ndc[1].xy(), coord),
            ) / self.det
        });
         */
        let det = f32x4::splat(self.det);
        let [
            [bx0, bx1, bx2, bx3],
            [by0, by1, by2, by3],
            [bz0, bz1, bz2, bz3]
        ] = [
            wide_det3(coords4, self.ndc[1].xy().into(), self.ndc[2].xy().into()) / det,
            wide_det3(self.ndc[0].xy().into(), coords4, self.ndc[2].xy().into()) / det,
            wide_det3(self.ndc[0].xy().into(), self.ndc[1].xy().into(), coords4) / det,
        ].map(f32x4::to_array);
        let barycentric = [
            Vec3::new(bx0, by0, bz0),
            Vec3::new(bx1, by1, bz1),
            Vec3::new(bx2, by2, bz2),
            Vec3::new(bx3, by3, bz3)
        ];
        if barycentric.iter().all(|b| b.cmplt(Vec3::ZERO).any()) {
            discard4!();
        }
        let perspective = barycentric.map(|barycentric| {
            // perspective correct interpolation
            // see: https://stackoverflow.com/questions/24441631/how-exactly-does-opengl-do-perspectively-correct-linear-interpolation
            let depths = [
                self.vertex_outputs[0].position().zw(),
                self.vertex_outputs[1].position().zw(),
                self.vertex_outputs[2].position().zw(),
            ];
            let barycentric_depth = Interpolate3::interpolate3(&depths, barycentric);
            (
                barycentric_depth.x,
                barycentric * (barycentric_depth.y / Vec3::new(depths[0].y, depths[1].y, depths[2].y)),
            )
        });
        if perspective.iter().all(|d| d.0 < 0. || d.0 > 1.) {
            discard4!();
        }
        let mut context = task::Context::from_waker(task::Waker::noop());
        let mut threads = [
            (perspective[0].1, &derivatives[0]),
            (perspective[1].1, &derivatives[1]),
            (perspective[2].1, &derivatives[2]),
            (perspective[3].1, &derivatives[3]),
        ]
        .map(|(perspective, derivative)| {
            Box::pin(async move {
                let input = Interpolate3::interpolate3(&self.vertex_outputs, perspective);
                let ndc = Interpolate3::interpolate3(&self.ndc, perspective);
                self.shader
                    .fragment(input, ndc, perspective, self.det >= 0., |x| derivative.get_result(x))
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
                    self.ndc.iter().fold(-Vec2::ONE, |acc, v| acc.min(v.xy())),
                    pixels,
                ),
                hi: ndc_to_pixel(
                    self.ndc.iter().fold(Vec2::ONE, |acc, v| acc.max(v.xy())),
                    pixels,
                ),
            }
        }
    }
}

/// Vertex shader outputs must have a clip space position
pub trait HasPosition {
    fn position(&self) -> Vec4;
    fn ndc(&self) -> Vec4 {
        let p = self.position();
        Vec4::new(p.x / p.w, p.y / p.w, 1., 1. / p.w)
    }
}

/// Define vertex and fragment shaders
pub trait Shader: Sized + Send + Sync {
    type Vertex;
    type VertexOutput: HasPosition + Interpolate3 + Clone + Send + Sync;
    type FragmentOutput;
    type Error;
    type DerivativeType: Default
        + Debug
        + Copy
        + Sub<Self::DerivativeType, Output = Self::DerivativeType>
        + Div<f32, Output = Self::DerivativeType>;
    fn vertex(&self, vertex: &Self::Vertex) -> Self::VertexOutput;
    fn fragment<F>(
        &self,
        input: Self::VertexOutput,
        ndc: Vec4,
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
