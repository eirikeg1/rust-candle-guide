//! Synthetic data generators for ML exercises.

use candle_core::{Device, Tensor};

/// Generate noisy linear data: `y = slope * x + intercept + noise`.
///
/// Returns `(x, y)` where both have shape `[n, 1]`.
pub fn linear_data(
    n: usize,
    slope: f32,
    intercept: f32,
    noise_std: f32,
    x_range: (f32, f32),
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_vals: Vec<f32> = (0..n)
        .map(|i| x_range.0 + i as f32 * (x_range.1 - x_range.0) / n as f32)
        .collect();

    let x = Tensor::new(x_vals.as_slice(), device)?.reshape((n, 1))?;
    let noise = Tensor::randn(0f32, noise_std, (n, 1), device)?;
    let y = ((&x * slope as f64)?.broadcast_add(&Tensor::new(intercept, device)?)? + noise)?;

    Ok((x, y))
}

/// Generate noisy sine data: `y = sin(x) + noise`.
///
/// Returns `(x, y)` where both have shape `[n, 1]`.
pub fn sine_data(
    n: usize,
    noise_std: f32,
    x_range: (f32, f32),
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let x_vals: Vec<f32> = (0..n)
        .map(|i| x_range.0 + i as f32 * (x_range.1 - x_range.0) / n as f32)
        .collect();
    let y_vals: Vec<f32> = x_vals.iter().map(|&x| x.sin()).collect();

    let x = Tensor::new(x_vals.as_slice(), device)?.reshape((n, 1))?;
    let y_clean = Tensor::new(y_vals.as_slice(), device)?.reshape((n, 1))?;
    let noise = Tensor::randn(0f32, noise_std, (n, 1), device)?;
    let y = (&y_clean + &noise)?;

    Ok((x, y))
}

/// Generate noisy XOR data with `n_per_corner` samples around each of the 4 corners.
///
/// Returns `(x, y)` where x has shape `[4*n_per_corner, 2]` and y has shape
/// `[4*n_per_corner]` with values in {0, 1}.
pub fn xor_data(
    n_per_corner: usize,
    noise_std: f32,
    device: &Device,
) -> anyhow::Result<(Tensor, Tensor)> {
    let centers: [(f32, f32, u32); 4] = [
        (0.0, 0.0, 0),
        (0.0, 1.0, 1),
        (1.0, 0.0, 1),
        (1.0, 1.0, 0),
    ];

    let n_total = n_per_corner * 4;
    let mut xs = Vec::with_capacity(n_total * 2);
    let mut ys = Vec::with_capacity(n_total);

    for &(cx, cy, label) in &centers {
        for _ in 0..n_per_corner {
            xs.push(cx);
            xs.push(cy);
            ys.push(label);
        }
    }

    let x_base = Tensor::new(xs.as_slice(), device)?.reshape((n_total, 2))?;
    let noise = Tensor::randn(0f32, noise_std, (n_total, 2), device)?;
    let x = (&x_base + &noise)?;
    let y = Tensor::new(ys.as_slice(), device)?;

    Ok((x, y))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear_data_shapes() {
        let device = Device::Cpu;
        let (x, y) = linear_data(100, 3.0, 2.0, 0.1, (0.0, 5.0), &device).unwrap();
        assert_eq!(x.dims(), &[100, 1]);
        assert_eq!(y.dims(), &[100, 1]);
    }

    #[test]
    fn test_sine_data_shapes() {
        let device = Device::Cpu;
        let pi = std::f32::consts::PI;
        let (x, y) = sine_data(200, 0.05, (-pi, pi), &device).unwrap();
        assert_eq!(x.dims(), &[200, 1]);
        assert_eq!(y.dims(), &[200, 1]);
    }

    #[test]
    fn test_xor_data_shapes() {
        let device = Device::Cpu;
        let (x, y) = xor_data(50, 0.2, &device).unwrap();
        assert_eq!(x.dims(), &[200, 2]);
        assert_eq!(y.dims(), &[200]);
    }

    #[test]
    fn test_xor_data_labels() {
        let device = Device::Cpu;
        let (_, y) = xor_data(10, 0.0, &device).unwrap();
        let labels = y.to_vec1::<u32>().unwrap();
        // First 10: class 0, next 10: class 1, next 10: class 1, last 10: class 0
        assert!(labels[..10].iter().all(|&l| l == 0));
        assert!(labels[10..20].iter().all(|&l| l == 1));
        assert!(labels[20..30].iter().all(|&l| l == 1));
        assert!(labels[30..].iter().all(|&l| l == 0));
    }
}
