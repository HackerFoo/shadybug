use core::f32::consts::PI;

use glam::{Mat4, Vec3, Vec4, Vec4Swizzles};
use image::{DynamicImage, Rgba, Rgba32FImage};
use shadybug::{
    DerivativeCell, HasColor, HasDepth, HasPosition, Interpolate3, SamplerError, Shader, discard,
    render,
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

    // the image will be 1024x1024
    let img_size = 1024u32;

    let mut image = Rgba32FImage::new(img_size, img_size);

    let indices: Vec<usize> = (0..36).collect();
    let vertices: Vec<Vertex> = CUBE_POSITIONS
        .iter()
        .map(|p| Vertex {
            position: (*p).into(),
        })
        .collect();

    // render to the image
    render(img_size, &bindings, &vertices, &indices, |x, y, color| {
        image.put_pixel(x, y, Rgba(color))
    });

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

/// Fragment output
#[allow(dead_code)]
#[derive(Debug)]
struct FragmentOutput {
    pub depth: f32,
    pub color: [f32; 4],
    pub world_position: Vec3,
    pub world_normal: Vec3,
}

impl HasDepth for FragmentOutput {
    fn depth(&self) -> f32 {
        self.depth
    }
}
impl HasColor for FragmentOutput {
    fn color(&self) -> [f32; 4] {
        self.color
    }
}

impl<'a> Shader for Bindings<'a> {
    type Vertex = Vertex;
    type VertexOutput = VertexOutput;
    type FragmentOutput = FragmentOutput;
    type Error = ();
    type DerivativeType = Vec3;
    fn vertex(&self, vertex: &Vertex) -> VertexOutput {
        let world_position = self.world_from_local * vertex.position.extend(1.);
        VertexOutput {
            position: self.view.clip_from_world * world_position,
            world_position: world_position.xyz() / world_position.w,
        }
    }
    async fn fragment(
        &self,
        input: VertexOutput,
        _ndc: Vec4,
        barycentric: Vec3,
        front_facing: bool,
        derivative: &DerivativeCell<Vec3>,
    ) -> Result<FragmentOutput, SamplerError> {
        if !front_facing {
            discard!();
        }

        // compute the world normal from the derivative of the world position
        let dpdx = derivative.dpdx(input.world_position).await?;
        let dpdy = derivative.dpdy(input.world_position).await?;
        let world_normal = dpdx.cross(dpdy).normalize();

        // simple lighting based on world normal
        let brightness = world_normal.z.max(0.).powi(2) * 0.8 + 0.2;
        let color = if barycentric.min_element() < 0.03 || (0.025..0.075).contains(&(barycentric.max_element() % 0.1)) {
            [0.1, 0.1, 0.1, 1.]
        } else {
            [brightness, 0., 0., 1.]
        };

        Ok(FragmentOutput {
            depth: input.position.z / input.position.w,
            color,
            world_position: input.world_position,
            world_normal,
        })
    }
}
