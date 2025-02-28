use glam::{UVec2, Vec2};

use crate::{Sampler, Shader};

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
