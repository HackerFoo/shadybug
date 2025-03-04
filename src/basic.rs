use std::ops::{Index, IndexMut};

use glam::{UVec2, Vec3, Vec4, Vec4Swizzles};

#[cfg(feature = "png")]
use std::{path::Path, fs::File, io::BufWriter};

#[derive(Clone, Debug)]
pub struct Target<Pixel> {
    pub size: UVec2,
    pub pixels: Vec<Pixel>,
}

impl<Pixel> Target<Pixel> {
    pub fn new(size: UVec2, pixel: Pixel) -> Self
    where
        Pixel: Clone,
    {
        Self {
            size,
            pixels: vec![pixel; size.element_product() as usize],
        }
    }
}

// from https://gamedev.stackexchange.com/a/194038
pub fn linear_to_srgb(color: Vec4) -> Vec4 {
    let rgb = color.xyz();
    let cutoff = rgb.cmplt(Vec3::splat(0.0031308));
    let higher = 1.055 * rgb.powf(1. / 2.4) - 0.055;
    let lower = rgb * 12.92;
    Vec3::select(cutoff, lower, higher).extend(color.w)
}

#[cfg(feature = "png")]
impl Target<Vec4> {
    pub fn write_png<P: AsRef<Path>>(self, path: P) {
        let file = File::create(path).unwrap();
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
        let mut writer = encoder.write_header().unwrap();
        let data: Vec<u8> = self
            .pixels
            .into_iter()
            .flat_map(|color| {
                linear_to_srgb(color)
                    .to_array()
                    .map(|x| (x * 256.).round().clamp(0., 255.) as u8)
            })
            .collect();
        writer.write_image_data(&data).unwrap();
    }
}

impl<Pixel> Index<UVec2> for Target<Pixel> {
    type Output = Pixel;

    fn index(&self, index: UVec2) -> &Self::Output {
        &self.pixels[(index.y * self.size.x + index.x) as usize]
    }
}

impl<Pixel> IndexMut<UVec2> for Target<Pixel> {
    fn index_mut(&mut self, index: UVec2) -> &mut Self::Output {
        &mut self.pixels[(index.y * self.size.x + index.x) as usize]
    }
}

#[derive(Default, Clone, Copy, Debug)]
pub struct Sample {
    pub depth: f32,
    pub color: Vec4,
}
