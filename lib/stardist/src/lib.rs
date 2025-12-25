use core::f32;
use ndarray::Zip;
use ndarray::{s, Array2, ArrayView3};
use numpy::ndarray::Array3;
use numpy::npyffi::npy_bool;
use numpy::{IntoPyArray, PyArray3, PyReadonlyArray3};
use pyo3::prelude::*;
use std::f32::consts::PI;

fn lower_bound_rays(
    src: &ArrayView3<npy_bool>,
    src_any: &Array2<bool>,
    rays: &[(f32, f32, f32)],
) -> Array3<f32> {
    let (n, height, width) = src.dim();

    let mut result = Array3::<f32>::from_elem((rays.len(), height, width), f32::NAN);

    Zip::indexed(&mut result).par_for_each(|(k, y, x), elem| {
        if !src_any[(y, x)] {
            return;
        }

        let (dx, dy, t_corr) = rays[k];
        *elem = f32::INFINITY;

        for i in 0..n {
            if src[(i, y, x)] != 0 {
                let mut x_ray = 0.0;
                let mut y_ray = 0.0;

                // Ray casting loop
                loop {
                    x_ray += dx;
                    y_ray += dy;
                    let xx = x_ray.round() as isize + x as isize;
                    let yy = y_ray.round() as isize + y as isize;

                    if xx < 0
                        || yy < 0
                        || yy as usize >= height
                        || xx as usize >= width
                        || src[(i, yy as usize, xx as usize)] == 0
                    {
                        // Small correction as we overshoot the boundary.
                        x_ray += t_corr * dx;
                        y_ray += t_corr * dy;

                        *elem = elem.min(x_ray.hypot(y_ray));
                        break;
                    }
                }
            }
        }
    });

    result
}

fn upper_bound_rays(
    src: &ArrayView3<npy_bool>,
    src_any: &Array2<bool>,
    rays: &[(f32, f32, f32)],
) -> Array3<f32> {
    let (_, height, width) = src.dim();

    let mut result = Array3::<f32>::from_elem((rays.len(), height, width), f32::NAN);

    Zip::indexed(&mut result).par_for_each(|(k, y, x), elem| {
        if src_any[(y, x)] {
            let (dx, dy, t_corr) = rays[k];
            let mut x_ray = 0.0;
            let mut y_ray = 0.0;

            loop {
                x_ray += dx;
                y_ray += dy;
                let xx = x_ray.round() as isize + x as isize;
                let yy = y_ray.round() as isize + y as isize;

                if xx < 0 || yy < 0 || yy as usize >= height || xx as usize >= width {
                    *elem = f32::INFINITY;
                    break;
                }

                if !src_any[(yy as usize, xx as usize)] {
                    // Small correction as we overshoot the boundary.
                    x_ray += t_corr * dx;
                    y_ray += t_corr * dy;

                    *elem = x_ray.hypot(y_ray);
                    break;
                }
            }
        }
    });

    result
}

#[pyfunction]
fn star_distances<'py>(
    py: Python<'py>,
    src: PyReadonlyArray3<'py, npy_bool>,
    n_rays: usize,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Bound<'py, PyArray3<f32>>)> {
    let src = src.as_array();
    let (_, height, width) = src.dim();

    // Precompute ray directions.
    let angle_step = (2.0 * PI) / n_rays as f32;
    let rays: Vec<(f32, f32, f32)> = (0..n_rays)
        .map(|k| {
            let phi = k as f32 * angle_step;
            let dx = phi.sin();
            let dy = phi.cos();
            let t_corr = 0.5 / f32::max(dx.abs(), dy.abs()) - 1.0;
            (dx, dy, t_corr)
        })
        .collect();

    let mut src_any = Array2::<bool>::from_elem((height, width), false);
    Zip::indexed(&mut src_any).par_for_each(|(y, x), elem| {
        *elem = src.slice(s![.., y, x]).iter().any(|&v| v != 0);
    });
    let lower = lower_bound_rays(&src, &src_any, &rays);
    let upper = upper_bound_rays(&src, &src_any, &rays);

    Ok((lower.into_pyarray(py), upper.into_pyarray(py)))
}

#[pymodule]
fn stardist(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(star_distances, m)?)?;
    Ok(())
}
