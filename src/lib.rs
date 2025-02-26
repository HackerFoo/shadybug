use std::{
    any::Any,
    collections::HashMap,
    ops::{Add, Div, Mul, Sub},
};

use glam::{UVec2, Vec2, Vec3, Vec4, Vec4Swizzles};

#[macro_export]
macro_rules! discard {
    () => {
        return Err(SamplerError::Discard.into())
    };
}

/// used to sample one pixel at a time to get fragment output
pub struct Sampler<'a, T: Shader> {
    pub shader: &'a T,
    pub vertex_outputs: [T::VertexOutput; 3],
    pub ndc: [Vec4; 3],
    pub det: f32,
}

/// determinate of AB and AC, gives twice the area of the triangle ABC
fn det3(a: Vec2, b: Vec2, c: Vec2) -> f32 {
    (b - a).perp_dot(c - a)
}

impl<'a, T: Shader> Sampler<'a, T> {
    /// build a sampler from a shader and a triangle
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

/// define how to interpolate using barycentric coordinates (weights.)
pub trait Interpolate3 {
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self
    where
        Self: Sized;
}

/// Used within shaders to implement partial derivatives
pub struct Diff {
    pub stage: Stage,
    pub sample_offset: Vec2,
    pub samples: HashMap<(usize, Stage), Box<dyn Any>>,
    pub id_offset: usize,
}

impl Diff {
    fn new() -> Self {
        Self {
            stage: Stage::Normal,
            sample_offset: Vec2::splat(0.05),
            samples: HashMap::new(),
            id_offset: 0,
        }
    }
    /// if using dpdx/dpdy in multiple function calls, use this to ensure unique ids
    pub fn id_offset(&mut self, id_offset: usize) -> &mut Self {
        self.id_offset = id_offset;
        self
    }
    fn store<V: Clone + 'static>(
        &mut self,
        value: V,
        mut id: usize,
        stage: Stage,
    ) -> Result<V, SamplerError> {
        id += self.id_offset;
        match self.stage {
            Stage::Normal => self
                .samples
                .get(&(id, stage))
                .ok_or(SamplerError::MissingSample)?
                .downcast_ref()
                .cloned()
                .ok_or(SamplerError::MissingSample),
            _ => {
                self.samples
                    .insert((id, self.stage), Box::new(value.clone()));
                Ok(value)
            }
        }
    }
    /// calculate the partial derivative with respect to X position
    /// id + id_offset must be unique per call within the fragment shader.
    pub fn dpdx<V>(&mut self, value: V, id: usize) -> Result<V, SamplerError>
    where
        V: Sub<V, Output = V> + Div<f32, Output = V> + Clone + 'static,
    {
        Ok((self.store(value.clone(), id, Stage::SampleX)? - value) / self.sample_offset.x)
    }
    /// calculate the partial derivative with respect to Y position
    /// id + id_offset must be unique per call within the fragment shader.
    pub fn dpdy<V>(&mut self, value: V, id: usize) -> Result<V, SamplerError>
    where
        V: Sub<V, Output = V> + Div<f32, Output = V> + Clone + 'static,
    {
        Ok((self.store(value.clone(), id, Stage::SampleY)? - value) / self.sample_offset.y)
    }
}

impl<T> Interpolate3 for T
where T: Add<T, Output = T> + Mul<f32, Output = T> + Copy
{
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self {
        input[0] * barycentric.x + input[1] * barycentric.y + input[2] * barycentric.z
    }
}

/// convert a pixel coordinate to normalized device coordinates
pub fn pixel_to_ndc(coord: UVec2, pixels: u32) -> Vec2 {
    let pixels = pixels as f32;
    coord.as_vec2() * (2. / pixels) - 1. + (0.5 / pixels)
}

/// convert normalized device coordinates to a pixel
fn ndc_to_pixel(coord: Vec2, pixels: u32) -> UVec2 {
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

/// upper and lower bounds for something
#[derive(Debug, Eq, PartialEq, Clone, Copy, Default)]
pub enum Bounds<A> {
    #[default]
    Zero,
    Bounds {
        lo: A,
        hi: A,
    },
}

impl<'a, T: Shader> Sampler<'a, T> {
    /// run the fragment shader for the given coordinates
    pub fn get(&self, coord: Vec2) -> Result<T::FragmentOutput, T::Error> {
        let mut result = None;
        let mut diff = Diff::new();
        for (stage, coord, keep) in [
            (Stage::SampleX, coord + diff.sample_offset.with_y(0.), false),
            (Stage::SampleY, coord + diff.sample_offset.with_x(0.), false),
            (Stage::Normal, coord, true),
        ] {
            diff.stage = stage;
            let barycentric = Vec3::new(
                det3(coord, self.ndc[1].xy(), self.ndc[2].xy()),
                det3(self.ndc[0].xy(), coord, self.ndc[2].xy()),
                det3(self.ndc[0].xy(), self.ndc[1].xy(), coord),
            ) / self.det;
            if keep && barycentric.cmplt(Vec3::ZERO).any() {
                discard!();
            }
            let depths = [
                self.vertex_outputs[0].position().zw(),
                self.vertex_outputs[1].position().zw(),
                self.vertex_outputs[2].position().zw(),
            ];
            let barycentric_depth = Interpolate3::interpolate3(&depths, barycentric);
            if barycentric_depth.x < 0. || barycentric_depth.x > 1. {
                discard!();
            }
            let perspective = barycentric * Vec3::new(depths[0].y, depths[1].y, depths[2].y)
                / barycentric_depth.y;
            let input = Interpolate3::interpolate3(&self.vertex_outputs, perspective);
            let ndc = Interpolate3::interpolate3(&self.ndc, perspective);
            result = Some(self.shader.fragment(input, ndc, self.det < 0., &mut diff));
        }
        result.unwrap()
    }
    /// calculate the bounds in pixels
    pub fn bounds(&self, pixels: u32) -> Bounds<UVec2> {
        if self.det >= 0. {
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

/// vertex shader outputs must have a clip space position
pub trait HasPosition {
    fn position(&self) -> Vec4;
    fn ndc(&self) -> Vec4 {
        let p = self.position();
        p.xyz().extend(1.) / p.w
    }
}

/// define vertex and fragment shaders
pub trait Shader: Sized {
    type Vertex;
    type VertexOutput: HasPosition + Interpolate3;
    type FragmentOutput;
    type Error: From<SamplerError>;
    fn vertex(&self, vertex: &Self::Vertex) -> Self::VertexOutput;
    fn fragment(
        &self,
        input: Self::VertexOutput,
        ndc: Vec4,
        front_facing: bool,
        diff: &mut Diff,
    ) -> Result<Self::FragmentOutput, Self::Error>;
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
}

/// stage of fragment shader, used to compute derivates
#[derive(PartialEq, Eq, Hash, Clone, Copy)]
pub enum Stage {
    SampleX,
    SampleY,
    Normal,
}

/// invalid fragment shader outputs
#[derive(Debug)]
pub enum SamplerError {
    Discard,
    MissingSample,
}
