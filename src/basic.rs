/// Basic types useful with the Shader trait
use std::ops::{Index, IndexMut};

use glam::{UVec2, Vec3, Vec4, Vec4Swizzles};

#[cfg(feature = "png")]
use std::{fs::File, io::BufWriter, path::Path};

#[derive(Clone, Debug)]
pub struct Target<Pixel> {
    pub size: UVec2,
    pub pixels: Vec<Pixel>,
}

impl<Pixel> Target<Pixel> {
    /// Create a new Target with the given `size` and fill it with `pixel`.
    pub fn new(size: UVec2, pixel: Pixel) -> Self
    where
        Pixel: Clone,
    {
        Self {
            size,
            pixels: vec![pixel; size.element_product() as usize],
        }
    }
    /// Flip the pixels vertically.
    pub fn flip_vertically(&mut self) {
        for y in 0..(self.size.y / 2) {
            let flipped_y = self.size.y - 1 - y;
            let low_row = y * self.size.x;
            let high_row = flipped_y * self.size.x;
            for x in 0..self.size.x {
                self.pixels.swap((low_row + x) as usize, (high_row + x) as usize);
            }
        }
    }
}

/// Convert from linear RGB to sRGB
/// from https://gamedev.stackexchange.com/a/194038
pub fn linear_to_srgb(color: Vec4) -> Vec4 {
    let rgb = color.xyz();
    let cutoff = rgb.cmplt(Vec3::splat(0.0031308));
    let higher = 1.055 * rgb.powf(1. / 2.4) - 0.055;
    let lower = rgb * 12.92;
    Vec3::select(cutoff, lower, higher).extend(color.w)
}

#[cfg(feature = "png")]
impl Target<Vec4> {
    /// Write target data as a PNG
    pub fn write_png<P: AsRef<Path>>(self, path: P) -> bool {
        // from png crate documentation
        let Ok(file) = File::create(path) else {
            return false;
        };
        let ref mut w = BufWriter::new(file);
        let mut encoder = png::Encoder::new(w, self.size.x, self.size.y);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder.set_source_gamma(png::ScaledFloat::new(1. / 2.2));
        let source_chromaticities = png::SourceChromaticities::new(
            (0.31270, 0.32900),
            (0.64000, 0.33000),
            (0.30000, 0.60000),
            (0.15000, 0.06000),
        );
        encoder.set_source_chromaticities(source_chromaticities);
        let Ok(mut writer) = encoder.write_header() else {
            return false;
        };

        // convert float pixels in linear RGBA to 8-bit sRGBA
        let data: Vec<u8> = self
            .pixels
            .into_iter()
            .flat_map(|color| {
                linear_to_srgb(color)
                    .to_array()
                    .map(|x| (x * 256.).round().clamp(0., 255.) as u8)
            })
            .collect();

        writer.write_image_data(&data).is_ok()
    }
}

impl<Pixel> Index<UVec2> for Target<Pixel> {
    type Output = Pixel;

    fn index(&self, index: UVec2) -> &Self::Output {
        let flipped_y = self.size.y - 1 - index.y;
        &self.pixels[(flipped_y * self.size.x + index.x) as usize]
    }
}

impl<Pixel> IndexMut<UVec2> for Target<Pixel> {
    fn index_mut(&mut self, index: UVec2) -> &mut Self::Output {
        let flipped_y = self.size.y - 1 - index.y;
        &mut self.pixels[(flipped_y * self.size.x + index.x) as usize]
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Sample {
    pub depth: f32,
    pub color: Vec4,
}
