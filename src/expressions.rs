use std::collections::HashMap;

use crate::complex::{Complex, Weighted};
use crate::complex_mapping::{compute_barycenters, compute_maps_svd_copies};
use crate::complex_mapping::{compute_map_svd, PreMappable};
use crate::complex_opt::{Interpolate, OptComplex, WeightedOptComplex};
use crate::complex_tensor::{TensorComplex, WeightedTensorComplex};
use crate::io::{
    iter_complex_f32, iter_complex_f64, iter_vert_simp_f32, iter_vert_simp_f64,
    iter_vert_simp_weight_f32, iter_vert_simp_weight_f64, iter_weighted_complex_f32,
    iter_weighted_complex_f64,
};
use crate::tensorwect::{ECTParams, TensorEct, TensorWect};
use crate::utils::{array2_to_tensor, tensor_to_flat};
use crate::{iter_complex, iter_vert_simp, iter_vert_simp_weight, iter_weighted_complex};
use ndarray::{Array2, ArrayView2};
use serde::Deserialize;

use pyo3_polars::derive::polars_expr;

use polars::prelude::*;
// use pyo3_polars::export::polars_core::prelude::*;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[polars_expr(output_type_func=same_output_type)]
pub fn barycenters(inputs: &[Series]) -> PolarsResult<Series> {
    iter_vert_simp!(&inputs[0], &inputs[1], |va, sa| {
        compute_barycenters(va, sa).into_raw_vec()
    })
}

#[derive(Clone, Copy, Deserialize)]
struct MapsSvdArgs {
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
}

/// Compute the SVD maps for a given complex (given by verts, simps weights)
#[polars_expr(output_type_func=same_output_type)] // TODO: when unflattened, need to change output
pub fn map_svd(inputs: &[Series], kwargs: MapsSvdArgs) -> PolarsResult<Series> {
    iter_vert_simp_weight!(&inputs[0], &inputs[1], &inputs[2], |va, sa, w| {
        let map = compute_map_svd(
            // TODO: unhardcode f32
            &va.view(),
            &sa.view(),
            w,
            kwargs.subsample_ratio,
            kwargs.subsample_min,
            kwargs.subsample_max,
        );

        // flatten
        map.into_raw_vec()
    })
}

#[derive(Clone, Copy, Deserialize)]
struct MapsSvdCopyArgs {
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f64>,
    copies: bool,
}

/// Compute the SVD maps for a given complex (given by verts, simps weights),
/// generating rotated + reflected copies of the Vt matrix.
#[polars_expr(output_type_func=same_output_type)] // TODO: when unflattened, need to change output
pub fn maps_svd_copies(inputs: &[Series], kwargs: MapsSvdCopyArgs) -> PolarsResult<Series> {
    iter_vert_simp_weight_f32(&inputs[0], &inputs[1], &inputs[2], |va, sa, w| {
        let maps: Vec<Array2<f32>> = compute_maps_svd_copies(
            // TODO: unhardcode f32
            &va.view(),
            &sa.view(),
            w,
            kwargs.subsample_ratio,
            kwargs.subsample_min,
            kwargs.subsample_max,
            kwargs.eps,
            kwargs.copies,
        );

        // flatten
        maps.into_iter()
            .map(|x| x.into_raw_vec())
            .flatten()
            .collect()
    })
}

fn struct_use_weights(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[1];
    match field.dtype() {
        DataType::Struct(fields) => Ok(fields[0].clone()), // use type of vertex weights
        _ => unreachable!(),
    }
}

fn struct_use_verts(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    match field.dtype() {
        DataType::Struct(fields) => Ok(fields[0].clone()), // use type of vertices
        _ => unreachable!(),
    }
}

#[derive(Clone, Deserialize)]
struct PremappedCopyWectArgs {
    num_heights: i64,
    num_directions: i64,
    provided_weights: Vec<usize>, // the dimensions of weights provided, in order
    align_dimension: usize,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f64>,
    copies: bool,
}

// Compute the WECT for a given complex by first computing the premaps
// and then applying the WECT. Useful for generating all possible rotated WECTs.
#[polars_expr(output_type_func=struct_use_weights)]
pub fn premapped_copy_wect(
    inputs: &[Series],
    kwargs: PremappedCopyWectArgs,
) -> PolarsResult<Series> {
    tch::maybe_init_cuda();
    let device = tch::Device::cuda_if_available();

    let mut ep_per_dim: HashMap<i64, ECTParams> = HashMap::new();

    // Get the right tch Kind for the directions
    let fields = &inputs[0].struct_()?.fields_as_series();
    let vdtype = fields[0].dtype().leaf_dtype();
    let kind: tch::Kind = match vdtype {
        DataType::Float32 => tch::Kind::Float,
        DataType::Float64 => tch::Kind::Double,
        _ => polars_bail!(InvalidOperation: format!(
            "vertex dtype {vdtype} not supported for premapped_copy_wect"
        )),
    };

    // Some(kwargs.eps.unwrap() as f64),
    iter_weighted_complex!(
        &inputs[0],
        &inputs[1],
        kwargs.provided_weights.clone(),
        |complex| {
            complex.interpolate_missing_down();
            let pre_rots = complex.premap_copy(
                kwargs.align_dimension,
                kwargs.subsample_ratio,
                kwargs.subsample_min,
                kwargs.subsample_max,
                kwargs.eps,
                kwargs.copies,
            );

            let embedded_dimension = complex.get_vertices().shape()[1] as i64;

            if !ep_per_dim.contains_key(&embedded_dimension) {
                ep_per_dim.insert(
                    embedded_dimension,
                    ECTParams::new(
                        embedded_dimension,
                        kwargs.num_directions,
                        kwargs.num_heights,
                        device,
                        kind,
                    ),
                );
            }

            let ep = ep_per_dim.get(&embedded_dimension).unwrap();

            let device = ep.dirs.device();
            let tensor_complex = WeightedTensorComplex::from(&complex, device);
            pre_rots // apply the WECT for each rotation, and flatten the output
                .iter()
                .map(|x| {
                    let tx = array2_to_tensor(x, device);
                    let wect = tensor_complex.pre_rot_wect(ep, tx);
                    tensor_to_flat(&wect, kind)
                })
                .collect::<Vec<Vec<_>>>()
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
        }
    )
}

#[derive(Clone, Deserialize)]
struct PremappedWectArgs {
    num_heights: i64,
    num_directions: i64,
    provided_weights: Vec<usize>, // the dimensions of weights provided, in order
    align_dimension: usize,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
}

// Compute the WECT for a given complex by first computing the premaps
// and then applying the WECT.
#[polars_expr(output_type_func=struct_use_weights)]
pub fn premapped_wect(inputs: &[Series], kwargs: PremappedWectArgs) -> PolarsResult<Series> {
    tch::maybe_init_cuda();
    let device = tch::Device::cuda_if_available();

    let mut ep_per_dim: HashMap<i64, ECTParams> = HashMap::new();

    // Get the right tch Kind for the directions
    let fields = &inputs[0].struct_()?.fields_as_series();
    let vdtype = fields[0].dtype().leaf_dtype();
    let kind: tch::Kind = match vdtype {
        DataType::Float32 => tch::Kind::Float,
        DataType::Float64 => tch::Kind::Double,
        _ => polars_bail!(InvalidOperation: format!(
            "vertex dtype {vdtype} not supported for premapped_wect"
        )),
    };

    iter_weighted_complex!(
        &inputs[0],
        &inputs[1],
        kwargs.provided_weights.clone(),
        |complex| {
            complex.interpolate_missing_down();
            let pre_rot = complex.premap(
                kwargs.align_dimension,
                kwargs.subsample_ratio,
                kwargs.subsample_min,
                kwargs.subsample_max,
            );

            let embedded_dimension = complex.get_vertices().shape()[1] as i64;

            if !ep_per_dim.contains_key(&embedded_dimension) {
                ep_per_dim.insert(
                    embedded_dimension,
                    ECTParams::new(
                        embedded_dimension,
                        kwargs.num_directions,
                        kwargs.num_heights,
                        device,
                        kind,
                    ),
                );
            }

            let ep = ep_per_dim.get(&embedded_dimension).unwrap();

            let device = ep.dirs.device();
            let tensor_complex = WeightedTensorComplex::from(&complex, device);

            let tx = array2_to_tensor(&pre_rot, device);
            let wect = tensor_complex.pre_rot_wect(ep, tx);
            tensor_to_flat(&wect, kind)
        }
    )
}

#[derive(Clone, Deserialize)]
struct WectArgs {
    num_heights: i64,
    num_directions: i64,
    provided_weights: Vec<usize>, // the dimensions of weights provided, in order
}

#[polars_expr(output_type_func=struct_use_weights)]
pub fn wect(inputs: &[Series], kwargs: WectArgs) -> PolarsResult<Series> {
    tch::maybe_init_cuda();
    let device = tch::Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let mut ep_per_dim: HashMap<i64, ECTParams> = HashMap::new();

    // Get the right tch Kind for the directions
    let fields = &inputs[0].struct_()?.fields_as_series();
    let vdtype = fields[0].dtype().leaf_dtype();
    let kind: tch::Kind = match vdtype {
        DataType::Float32 => tch::Kind::Float,
        DataType::Float64 => tch::Kind::Double,
        _ => polars_bail!(InvalidOperation: format!(
            "vertex dtype {vdtype} not supported for wect"
        )),
    };

    iter_weighted_complex!(
        &inputs[0],
        &inputs[1],
        kwargs.provided_weights.clone(),
        |complex| {
            complex.interpolate_missing_down();

            let embedded_dimension = complex.get_vertices().shape()[1] as i64;

            if !ep_per_dim.contains_key(&embedded_dimension) {
                ep_per_dim.insert(
                    embedded_dimension,
                    ECTParams::new(
                        embedded_dimension,
                        kwargs.num_directions,
                        kwargs.num_heights,
                        device,
                        kind,
                    ),
                );
            }

            let ep = ep_per_dim.get(&embedded_dimension).unwrap();

            let device = ep.dirs.device();
            let tensor_complex = WeightedTensorComplex::from(&complex, device);

            let wect: tch::Tensor = tensor_complex.wect(ep);
            tensor_to_flat(&wect, kind)
        }
    )
}

#[derive(Clone, Deserialize)]
struct EctArgs {
    num_heights: i64,
    num_directions: i64,
}

// Compute the ECT for a given complex
#[polars_expr(output_type_func=struct_use_verts)]
pub fn ect(inputs: &[Series], kwargs: EctArgs) -> PolarsResult<Series> {
    tch::maybe_init_cuda();
    let device = tch::Device::cuda_if_available();
    println!("Using device: {:?}", device);

    let mut ep_per_dim: HashMap<i64, ECTParams> = HashMap::new();

    // Get the right tch Kind for the directions
    let fields = &inputs[0].struct_()?.fields_as_series();
    let vdtype = fields[0].dtype().leaf_dtype();
    let kind: tch::Kind = match vdtype {
        DataType::Float32 => tch::Kind::Float,
        DataType::Float64 => tch::Kind::Double,
        _ => polars_bail!(InvalidOperation: format!(
            "vertex dtype {vdtype} not supported for ect"
        )),
    };

    iter_complex!(&inputs[0], |complex| {
        complex.interpolate_missing_down();
        let embedded_dimension = complex.get_vertices().shape()[1] as i64;

        if !ep_per_dim.contains_key(&embedded_dimension) {
            ep_per_dim.insert(
                embedded_dimension,
                ECTParams::new(
                    embedded_dimension,
                    kwargs.num_directions,
                    kwargs.num_heights,
                    device,
                    kind,
                ),
            );
        }

        let ep = ep_per_dim.get(&embedded_dimension).unwrap();
        let device = ep.dirs.device();
        let tensor_complex = TensorComplex::from(&complex, device);

        let ect: tch::Tensor = tensor_complex.ect(ep);
        tensor_to_flat(&ect, kind)
    })
}
