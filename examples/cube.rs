use std::f32::consts::PI;

use glam::{Mat4, UVec2, Vec3, Vec4, Vec4Swizzles};
use image::{DynamicImage, Rgba, Rgba32FImage};
use shadybug::{
    Bounds, Diff, HasPosition, Interpolate3, SamplerError, Shader, discard, pixel_to_ndc,
};

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

    // the image will be 512x512
    let img_size = 512u32;

    let mut depth_buffer = vec![0.; (img_size * img_size) as usize];
    let mut image = Rgba32FImage::new(img_size, img_size);

    let indices: Vec<usize> = (0..36).collect();
    let vertices: Vec<Vertex> = CUBE_POSITIONS
        .iter()
        .map(|p| Vertex {
            position: (*p).into(),
        })
        .collect();

    // for each triangle
    for indices in indices.chunks(3) {
        // draw_triangle runs the vertex shader for each vertex,
        // and builds a sampler that can run the fragment shader for each pixel
        let sampler = bindings.draw_triangle(&vertices, indices);

        // sample each pixel within the bounding box of the triangle
        if let Bounds::Bounds { lo, hi } = sampler.bounds(img_size) {
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let pixel = image.get_pixel_mut(x, y);
                    let depth = &mut depth_buffer[(y * img_size + x) as usize];
                    if let Ok(output) = sampler.get(pixel_to_ndc(UVec2::new(x, y), img_size)) {
                        if output.depth > *depth {
                            *depth = output.depth;
                            *pixel = Rgba(output.color);
                        }
                    }
                }
            }
        }
    }

    // convert to an 8-bit image and save
    DynamicImage::ImageRgba32F(image)
        .into_rgba8()
        .save("cube.png")
        .unwrap();
}

/// view transforms
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
    /// create a new view by computing from given values
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

/// bindings available to all triangles
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

/// vertex shader input
#[derive(Debug)]
struct Vertex {
    position: Vec3,
}

/// vertex shader output
/// these are interpolated within a triangle for fragment shader input
/// this must have a clip space position
#[derive(Debug)]
struct VertexOutput {
    position: Vec4,
    world_position: Vec3,
}

impl Interpolate3 for VertexOutput {
    fn interpolate3(input: &[Self; 3], barycentric: Vec3) -> Self {
        Self {
            position: input[0].position * barycentric.x
                + input[1].position * barycentric.y
                + input[2].position * barycentric.z,
            world_position: input[0].world_position * barycentric.x
                + input[1].world_position * barycentric.y
                + input[2].world_position * barycentric.z,
        }
    }
}

impl HasPosition for VertexOutput {
    fn position(&self) -> Vec4 {
        self.position
    }
}

/// fragment output
#[derive(Debug)]
struct FragmentOutput {
    pub depth: f32,
    pub color: [f32; 4],
    pub world_position: Vec3,
    pub world_normal: Vec3,
}

impl<'a> Shader for Bindings<'a> {
    type Vertex = Vertex;
    type VertexOutput = VertexOutput;
    type FragmentOutput = FragmentOutput;
    type Error = SamplerError;
    fn vertex(&self, vertex: &Vertex) -> VertexOutput {
        let world_position = self.world_from_local * vertex.position.extend(1.);
        VertexOutput {
            position: self.view.clip_from_world * world_position,
            world_position: world_position.xyz() / world_position.w,
        }
    }
    fn fragment(
        &self,
        input: VertexOutput,
        _ndc: Vec4,
        front_facing: bool,
        diff: &mut Diff,
    ) -> Result<FragmentOutput, SamplerError> {
        if !front_facing {
            discard!();
        }

        // compute the world normal from the derivative of the world position
        let world_normal = diff
            .dpdx(input.world_position, 0)?
            .cross(diff.dpdy(input.world_position, 0)?)
            .normalize();

        // simple lighting based on world normal
        let brightness = world_normal.z.max(0.) * 0.8 + 0.2;
        let mut color = [brightness, 0., 0., 1.];
        if world_normal.z > 0.5 {
            // color[2] = 1.;
        }

        Ok(FragmentOutput {
            depth: input.position.z / input.position.w,
            color,
            world_position: input.world_position,
            world_normal,
        })
    }
}
