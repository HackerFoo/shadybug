use core::f32::consts::PI;

use glam::*;
use image::{DynamicImage, Rgba, Rgba32FImage};
use shadybug::*;

const CUBE_POSITIONS: [[f32; 3]; 36] = [
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
    [-1.0, -1.0, 1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0],
    [1.0, 1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
];

// render a cube to cube.png
fn main() {
    // camera at (0, 0, 4) looking to the origin
    let view_from_world = Mat4::look_to_rh(Vec3::new(0., 0., 4.), -Vec3::Z, Vec3::Y);

    // 45 degree field of view, square aspect ratio, 0.1 near plane
    let clip_from_view = Mat4::perspective_infinite_reverse_rh(PI / 4., 1., 0.1);

    let view = View::new(view_from_world, clip_from_view);

    // rotate the cube and scale it a bit to make it look nice
    let world_from_local =
        Mat4::from_axis_angle(Vec3::ONE.normalize(), PI / 4.) * Mat4::from_scale(Vec3::splat(0.8));

    let bindings = Bindings::new(&view, world_from_local);

    // the image will be 1024x1024
    let img_size = 1024u32;

    let indices: Vec<usize> = (0..36).collect();
    let vertices: Vec<Vertex> = CUBE_POSITIONS
        .iter()
        .map(|p| Vertex {
            position: (*p).into(),
        })
        .collect();

    // render to the image
    let image = render(
        img_size,
        &bindings,
        &vertices,
        &indices,
        Rgba32FImage::new(img_size, img_size),
    );

    // convert to an 8-bit image and save
    DynamicImage::ImageRgba32F(image)
        .into_rgba8()
        .save("cube.png")
        .unwrap();
}

/// View transforms
#[derive(Debug)]
pub struct View {
    pub world_position: Vec3,
    pub world_from_view: Mat4,
    pub view_from_world: Mat4,
    pub clip_from_view: Mat4,
    pub view_from_clip: Mat4,
    pub clip_from_world: Mat4,
}

impl View {
    /// Create a new view by computing from given values
    pub fn new(view_from_world: Mat4, clip_from_view: Mat4) -> Self {
        let world_from_view = view_from_world.inverse();
        let world_position = world_from_view.w_axis.xyz() / world_from_view.w_axis.w;
        let view_from_clip = clip_from_view.inverse();
        let clip_from_world = clip_from_view * view_from_world;
        Self {
            world_position,
            world_from_view,
            view_from_world,
            clip_from_view,
            view_from_clip,
            clip_from_world,
        }
    }
}

/// Bindings available to all triangles
#[derive(Debug)]
struct Bindings<'a> {
    world_from_local: Mat4,
    view: &'a View,
}

impl<'a> Bindings<'a> {
    fn new(view: &'a View, world_from_local: Mat4) -> Self {
        Self {
            world_from_local,
            view,
        }
    }
}

/// Vertex shader input
#[derive(Debug)]
struct Vertex {
    position: Vec3,
}

/// Vertex shader output
/// These are interpolated within a triangle for fragment shader input.
/// This must have a clip space position.
#[derive(Debug, Clone, Copy)]
struct VertexOutput {
    position: Vec4,
    local_position: Vec3,
    world_position: Vec3,
}

impl Interpolate3 for VertexOutput {
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self {
        Self {
            position: Interpolate3::interpolate3(&input.map(|x| x.position), barycentric),
            local_position: Interpolate3::interpolate3(
                &input.map(|x| x.local_position),
                barycentric,
            ),
            world_position: Interpolate3::interpolate3(
                &input.map(|x| x.world_position),
                barycentric,
            ),
        }
    }
}

/// Fragment output
#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
struct FragmentOutput {
    pub depth: f32,
    pub color: Vec4,
    pub world_position: Vec3,
    pub world_normal: Vec3,
}

#[derive(Default, Clone, Copy, Debug)]
struct Sample {
    pub depth: f32,
    pub color: Vec4,
}

impl<'a> Shader for Bindings<'a> {
    type Vertex = Vertex;
    type VertexOutput = VertexOutput;
    type FragmentInput = VertexOutput;
    type FragmentOutput = FragmentOutput;
    type Sample = Sample;
    type Target = Rgba32FImage;
    type Error = ();
    type DerivativeType = Vec3;

    fn vertex(&self, vertex: &Vertex) -> VertexOutput {
        let world_position = self.world_from_local * vertex.position.extend(1.);
        VertexOutput {
            position: self.view.clip_from_world * world_position,
            local_position: vertex.position,
            world_position: world_position.xyz() / world_position.w,
        }
    }
    fn perspective(mut vertex_output: Self::VertexOutput) -> (Self::FragmentInput, Vec2, f32) {
        let inv_w = vertex_output.position.w.recip();
        vertex_output.position *= inv_w;
        vertex_output.local_position *= inv_w;
        vertex_output.world_position *= inv_w;
        (vertex_output, vertex_output.position.xy(), inv_w)
    }
    async fn fragment<F>(
        &self,
        input: VertexOutput,
        barycentric: Vec3,
        front_facing: bool,
        derivative: F,
    ) -> Result<FragmentOutput, SamplerError>
    where
        F: AsyncFn(Vec3) -> (Vec3, Vec3),
    {
        if !front_facing {
            discard!();
        }

        // compute the world normal from the derivative of the world position
        let (dpdx, dpdy) = derivative(input.world_position).await;
        let world_normal = dpdx.cross(dpdy).normalize();

        // simple lighting based on world normal
        let light = world_normal.z.max(0.) * 0.8 + 0.2;
        let lines = if barycentric.min_element() < 0.03
            || (0.025..0.075).contains(&(barycentric.max_element() % 0.1))
        {
            0.4
        } else {
            1.0
        };
        let checker = if matches!(
            ((input.local_position + 10.125) % 0.5)
                .cmpgt(Vec3::splat(0.25))
                .bitmask(),
            0 | 3 | 5 | 6
        ) {
            1.
        } else {
            0.75
        };
        let color = light * lines * checker * vec3(1., 0., 0.);

        Ok(FragmentOutput {
            depth: input.position.z / input.position.w,
            color: color.extend(1.),
            world_position: input.world_position,
            world_normal,
        })
    }
    fn combine(fragment: Self::FragmentOutput, pixel: &mut Self::Sample) -> bool {
        if fragment.depth > pixel.depth {
            pixel.depth = fragment.depth;
            pixel.color = fragment.color;
            true
        } else {
            false
        }
    }
    fn merge<I: Iterator<Item = ((UVec2, bool), Self::Sample)>>(
        offset: UVec2,
        size: UVec2,
        iter: I,
        target: &mut Self::Target,
    ) {
        // merge subtarget into target
        for ((pos, subsampled), sample) in iter {
            let w = if subsampled { 0.25 } else { 1. };
            let coord = offset + pos;
            let pixel = target.get_pixel_mut(coord.x, target.height() - 1 - coord.y);
            let color = Vec4::from(pixel.0) + sample.color * w;
            *pixel = Rgba(color.into());
        }

        // convert from pre-multiplied alpha
        for y in 0..size.y {
            for x in 0..size.x {
                let coord = offset + UVec2::new(x, y);
                let pixel = target.get_pixel_mut(coord.x, target.height() - 1 - coord.y);
                let color = Vec4::from(pixel.0);
                if color.w > 0. {
                    *pixel = Rgba((color.xyz() / color.w).extend(color.w).into());
                }
            }
        }
    }
}
