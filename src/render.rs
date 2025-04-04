use glam::{UVec2, Vec2};

use crate::{Sampler, Shader};

/// A tile with all samplers that cover it
pub struct SamplerTile<'a, T: Shader> {
    pub offset: UVec2,
    pub size: UVec2,
    pub pixels: u32,
    pub alignment: u32,
    pub samplers: Vec<Sampler<'a, T>>,
}

impl<'a, T: Shader<FragmentInput: Clone>> SamplerTile<'a, T> {
    /// Split the tile horizontally
    pub fn split_x(self) -> [Self; 2] {
        let mut mid = self.size.x / 2;
        mid += mid % self.alignment;
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
                alignment: self.alignment,
                samplers: left,
            },
            Self {
                offset: UVec2::new(self.offset.x + mid, self.offset.y),
                size: UVec2::new(self.size.x - mid, self.size.y),
                pixels: self.pixels,
                alignment: self.alignment,
                samplers: right,
            },
        ]
    }
    /// Split the tile vertically
    pub fn split_y(self) -> [Self; 2] {
        let mut mid = self.size.y / 2;
        mid += mid % self.alignment;
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
                alignment: self.alignment,
                samplers: bottom,
            },
            Self {
                offset: UVec2::new(self.offset.x, self.offset.y + mid),
                size: UVec2::new(self.size.x, self.size.y - mid),
                pixels: self.pixels,
                alignment: self.alignment,
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

impl<'a, T: Shader<FragmentInput: Clone>> Iterator for SamplerTileIter<'a, T> {
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

/// Simple tiled renderer
/// The `consume` function takes x and y coordinates, depth, and fragment output.
///  It will compare and set depth as well as consume the fragment output.
#[allow(unused_mut)]
pub fn render<S>(
    img_size: u32,
    bindings: &S,
    vertices: &[S::Vertex],
    indices: &[usize],
    mut target: S::Target,
) -> S::Target
where
    S: Shader<FragmentInput: Clone + Send + Sync, Target: Send + Sync> + Send + Sync,
{
    #[cfg(feature = "rayon")]
    use rayon::iter::{ParallelBridge, ParallelIterator};
    #[cfg(feature = "rayon")]
    use std::sync::{Arc, Mutex};

    const TILE_ALIGNMENT: u32 = 2;
    const TILE_SIZE: u32 = 32;
    const COORD_OFFSETS: [UVec2; 4] = [
        UVec2::new(0, 0),
        UVec2::new(1, 0),
        UVec2::new(0, 1),
        UVec2::new(1, 1),
    ];

    // compute values for all fragments
    let offsets = pixel_to_ndc(UVec2::splat(1), img_size) - pixel_to_ndc(UVec2::splat(0), img_size);
    let inverse_offsets = offsets.recip();
    let sample_offsets = [
        Vec2::new(0., 0.),
        Vec2::new(1., 0.),
        Vec2::new(0., 1.),
        Vec2::new(1., 1.),
    ]
    .map(|v| offsets * v);

    fn index(coord: UVec2, row_width: u32) -> usize {
        (coord.y * row_width + coord.x) as usize
    }

    // with rayon
    #[cfg(feature = "rayon")]
    let target_ref = Arc::new(Mutex::new(target));
    #[cfg(feature = "rayon")]
    let iter = bindings
        .tiled_iter(&vertices, &indices, img_size, TILE_ALIGNMENT, TILE_SIZE)
        .par_bridge();

    // without rayon
    #[cfg(not(feature = "rayon"))]
    let iter = bindings.tiled_iter(vertices, indices, img_size, TILE_ALIGNMENT, TILE_SIZE);

    // iterate over tiles
    iter.for_each(|tile| {
        let tile_area = tile.size.element_product();
        let mut subtarget = vec![Default::default(); tile_area as usize];
        for sampler in &tile.samplers {
            // sample each pixel within the bounding box of the triangle
            if let Bounds::Bounds { mut lo, mut hi } = tile.bounds(sampler) {
                for y in (lo.y..hi.y).step_by(2) {
                    for x in (lo.x..hi.x).step_by(2) {
                        let tile_coord = UVec2::new(x, y) - lo;
                        let coord = pixel_to_ndc(UVec2::new(x, y), img_size);
                        let samples = sampler.get(coord, &sample_offsets, inverse_offsets);
                        for (offset, output) in COORD_OFFSETS.into_iter().zip(samples.into_iter()) {
                            if let Ok(output) = output {
                                S::combine(
                                    output,
                                    &mut subtarget[index(tile_coord + offset, tile.size.x)],
                                );
                            }
                        }
                    }
                }
            }
        }

        // with rayon, clone and unlock
        #[cfg(feature = "rayon")]
        let target = Arc::clone(&target_ref);
        #[cfg(feature = "rayon")]
        let target = &mut *target.lock().unwrap();

        // without rayon, directly get a mutable reference
        #[cfg(not(feature = "rayon"))]
        let target = &mut target;

        let iter = (0..tile.size.y)
            .flat_map(move |y| (0..tile.size.x).map(move |x| UVec2::new(x, y)))
            .zip(subtarget);

        S::merge(tile.offset, tile.size, iter, target);
    });

    // with rayon, unwrap the Arc<Mutex<_>>
    #[cfg(feature = "rayon")]
    return Arc::into_inner(target_ref).unwrap().into_inner().unwrap();

    // without rayon
    #[cfg(not(feature = "rayon"))]
    return target;
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
