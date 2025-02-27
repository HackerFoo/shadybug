use std::{
    cell::Cell,
    fmt::Debug,
    future::Future,
    ops::{Add, Deref, Div, Mul, Sub},
    pin::Pin,
    task::{self, Context, Poll},
};

use glam::{UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};

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

/// Convert a pixel coordinate to normalized device coordinates
pub fn pixel_to_ndc(coord: UVec2, pixels: u32) -> Vec2 {
    let pixels = pixels as f32;
    coord.as_vec2() * (2. / pixels) - 1. + (0.5 / pixels)
}

/// Convert normalized device coordinates to a pixel
pub fn ndc_to_pixel(coord: Vec2, pixels: u32) -> UVec2 {
    let pixels = pixels as f32;
    ((coord - (0.5 / pixels) + 1.) * 0.5 * pixels).as_uvec2()
}

#[test]
#[cfg(test)]
fn test_ndc_pixel_conversion() {
    for x in 0..512 {
        let ndc = pixel_to_ndc(UVec2::splat(x), 512);
        assert!(ndc.x >= -1. && ndc.x <= 1.);
        assert_eq!(x, ndc_to_pixel(ndc, 512).x);
    }
    assert_eq!(ndc_to_pixel(Vec2::splat(-1.), 512).x, 0);
    assert_eq!(ndc_to_pixel(Vec2::splat(0.), 512).x, 255);
    assert_eq!(ndc_to_pixel(Vec2::splat(1.), 512).x, 511);
}

/// Upper and lower bounds for something
#[derive(Debug, Eq, PartialEq, Clone, Copy, Default)]
pub enum Bounds<A> {
    #[default]
    Zero,
    Bounds {
        lo: A,
        hi: A,
    },
}

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
        let barycentric = sample_offsets.map(|offset| {
            let coord = coord + offset;

            // barycentric coordinates, such that multiplying the weights by
            // the vertices will give the coordinates back in position.xy
            // limit to 1 to ensure coordinates will stay in bounds
            Vec3::new(
                det3(coord, self.ndc[1].xy(), self.ndc[2].xy()),
                det3(self.ndc[0].xy(), coord, self.ndc[2].xy()),
                det3(self.ndc[0].xy(), self.ndc[1].xy(), coord),
            ) / self.det
        });
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
                barycentric * Vec3::new(depths[0].y, depths[1].y, depths[2].y)
                    / barycentric_depth.y,
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
                    .fragment(input, ndc, self.det >= 0., derivative)
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
                if let Some(input) = src.get().get_input() {
                    *dst = input;
                } else {
                    return results.map(|r| r.unwrap_or(Err(SamplerError::MissingSample)));
                }
            }

            // calculate derivatives
            let dpdx = [(d[1] - d[0]) / offsets.x, (d[3] - d[2]) / offsets.x];
            let dpdy = [(d[2] - d[0]) / offsets.y, (d[3] - d[1]) / offsets.y];
            derivatives[0].set(Derivative::Output(dpdx[0], dpdy[0]));
            derivatives[1].set(Derivative::Output(dpdx[0], dpdy[1]));
            derivatives[2].set(Derivative::Output(dpdx[1], dpdy[0]));
            derivatives[3].set(Derivative::Output(dpdx[1], dpdy[1]));
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
        p.xyz().extend(1.) / p.w
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
    fn fragment(
        &self,
        input: Self::VertexOutput,
        ndc: Vec4,
        front_facing: bool,
        derivative: &DerivativeCell<Self::DerivativeType>,
    ) -> impl std::future::Future<Output = Result<Self::FragmentOutput, SamplerError<Self::Error>>> + Send;
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

/// A tile with all samplers that cover it
pub struct SamplerTile<'a, T: Shader> {
    pub offset: UVec2,
    pub size: UVec2,
    pub pixels: u32,
    pub samplers: Vec<Sampler<'a, T>>,
}

impl<'a, T: Shader> SamplerTile<'a, T> {
    /// Split the tile horizontally
    pub fn split_x(self) -> [Self; 2] {
        let mid = self.size.x / 2;
        let mut left = Vec::new();
        let mut right = Vec::new();
        for sampler in self.samplers {
            if let Bounds::Bounds { lo, hi } = sampler.bounds(self.pixels) {
                if mid > hi.x {
                    left.push(sampler);
                } else if mid <= lo.x {
                    right.push(sampler);
                } else {
                    left.push(sampler.clone());
                    right.push(sampler);
                }
            }
        }
        [
            Self {
                offset: self.offset,
                size: UVec2::new(mid, self.size.y),
                pixels: self.pixels,
                samplers: left,
            },
            Self {
                offset: UVec2::new(self.offset.x + mid, self.offset.y),
                size: UVec2::new(self.size.x - mid, self.size.y),
                pixels: self.pixels,
                samplers: right,
            },
        ]
    }
    /// Split the tile vertically
    pub fn split_y(self) -> [Self; 2] {
        let mid = self.size.y / 2;
        let mut bottom = Vec::new();
        let mut top = Vec::new();
        for sampler in self.samplers {
            if let Bounds::Bounds { lo, hi } = sampler.bounds(self.pixels) {
                if mid > hi.y {
                    bottom.push(sampler);
                } else if mid <= lo.y {
                    top.push(sampler);
                } else {
                    bottom.push(sampler.clone());
                    top.push(sampler);
                }
            }
        }
        [
            Self {
                offset: self.offset,
                size: UVec2::new(self.size.x, mid),
                pixels: self.pixels,
                samplers: bottom,
            },
            Self {
                offset: UVec2::new(self.offset.x, self.offset.y + mid),
                size: UVec2::new(self.size.x, self.size.y - mid),
                pixels: self.pixels,
                samplers: top,
            },
        ]
    }
    /// Bounds of the sampler intersecting the tile
    pub fn bounds(&self, sampler: &Sampler<'a, T>) -> Bounds<UVec2> {
        match sampler.bounds(self.pixels) {
            Bounds::Zero => Bounds::Zero,
            Bounds::Bounds { lo, hi } => {
                let lo = lo.max(self.offset);
                let hi = hi.min(self.offset + self.size);
                if lo.cmpgt(hi).any() {
                    Bounds::Zero
                } else {
                    Bounds::Bounds { lo, hi }
                }
            }
        }
    }
}

/// Iterator for tiles, splitting until they are no more than max_size
pub struct SamplerTileIter<'a, T: Shader> {
    pub max_size: u32,
    pub tiles: Vec<SamplerTile<'a, T>>,
}

impl<'a, T: Shader> Iterator for SamplerTileIter<'a, T> {
    type Item = SamplerTile<'a, T>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            let tile = self.tiles.pop()?;
            if tile.size.x >= tile.size.y && tile.size.x > self.max_size {
                for tile in tile.split_x() {
                    if !tile.samplers.is_empty() {
                        self.tiles.push(tile);
                    }
                }
            } else if tile.size.y > self.max_size {
                for tile in tile.split_y() {
                    if !tile.samplers.is_empty() {
                        self.tiles.push(tile);
                    }
                }
            } else {
                return Some(tile);
            }
        }
    }
}

/// Has a depth value
pub trait HasDepth {
    fn depth(&self) -> f32;
}

/// Has a color value
pub trait HasColor {
    fn color(&self) -> [f32; 4];
}

/// Simple tiled renderer
pub fn render<S, F>(
    img_size: u32,
    bindings: &S,
    vertices: &[S::Vertex],
    indices: &[usize],
    mut put_pixel: F,
) where
    S: Shader<FragmentOutput: HasDepth + HasColor>,
    F: FnMut(u32, u32, [f32; 4]),
{
    const TILE_SIZE: u32 = 32;
    let mut depth_buffer = vec![0.; (TILE_SIZE * TILE_SIZE) as usize];
    let offsets = pixel_to_ndc(UVec2::splat(1), img_size) - pixel_to_ndc(UVec2::splat(0), img_size);
    for tile in bindings.tiled_iter(&vertices, &indices, img_size, TILE_SIZE) {
        depth_buffer.fill(0.);
        for sampler in &tile.samplers {
            // sample each pixel within the bounding box of the triangle
            if let Bounds::Bounds { lo, hi } = tile.bounds(&sampler) {
                for y in (lo.y..hi.y).step_by(2) {
                    for x in (lo.x..hi.x).step_by(2) {
                        let coords = [(x, y), (x + 1, y), (x, y + 1), (x + 1, y + 1)];
                        for ((x, y), output) in coords.into_iter().zip(
                            sampler
                                .get(pixel_to_ndc(UVec2::new(x, y), img_size), offsets)
                                .into_iter(),
                        ) {
                            if let Ok(output) = output {
                                let depth = &mut depth_buffer
                                    [((y - lo.y) * TILE_SIZE + (x - lo.x)) as usize];
                                if output.depth() > *depth {
                                    *depth = output.depth();
                                    put_pixel(x, y, output.color());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
