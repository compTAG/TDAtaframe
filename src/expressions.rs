use std::collections::VecDeque;

use crate::complex::{WeightedOptComplex, WeightedTensorComplex};
use crate::complex_interpolation::Interpolate;
use crate::complex_mapping::PreMappable;
use crate::complex_mapping::{compute_barycenters, compute_maps_svd};
use crate::io::{iter_complex, iter_vert_simp, iter_vert_simp_weight};
use crate::tensorwect::{TensorWect, WECTParams};
use crate::utils::{array2_to_tensor, tensor_to_array2};
use ndarray::{Array2, ArrayView2};
use polars_core::utils::arrow::array::{Array, PrimitiveArray};
use serde::Deserialize;

use pyo3_polars::derive::polars_expr;

use pyo3_polars::export::polars_core::prelude::*;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

#[derive(Clone, Copy, Deserialize)]
struct BarycentersArgs {
    embedded_dimension: usize,
    simplex_dimension: usize,
}

#[polars_expr(output_type_func=same_output_type)] // TODO: Make generic
pub fn barycenters(inputs: &[Series], kwargs: BarycentersArgs) -> PolarsResult<Series> {
    iter_vert_simp(
        &inputs[0],
        &inputs[1],
        kwargs.embedded_dimension,
        kwargs.simplex_dimension,
        |va, sa| compute_barycenters(va, sa).into_raw_vec(),
    )
}

#[derive(Clone, Copy, Deserialize)]
struct MapsSvdArgs {
    embedded_dimension: usize,
    simplex_dimension: usize,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f32>,
    copies: bool,
}

#[polars_expr(output_type_func=same_output_type)] // TODO: when unflattened, need to change output
pub fn maps_svd(inputs: &[Series], kwargs: MapsSvdArgs) -> PolarsResult<Series> {
    iter_vert_simp_weight(
        &inputs[0],
        &inputs[1],
        &inputs[2],
        kwargs.embedded_dimension,
        kwargs.simplex_dimension,
        |va, sa, w| {
            let maps: Vec<Array2<f32>> = compute_maps_svd(
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
        },
    )
}

#[derive(Clone, Deserialize)]
struct PremappedWectArgs {
    embedded_dimension: i64,
    num_heights: i64,
    num_directions: i64,
    provided_simplices: Vec<usize>, // the dimensions of simplices provied, in order, starting with 1 // TODO: struct
    provided_weights: Vec<usize>,   // the dimensions of weights provided, in order
    align_dimension: usize,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f32>,
    copies: bool,
}

fn struct_use_weights(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[1];
    match field.data_type() {
        DataType::Struct(fields) => {
            Ok(fields[0].clone()) // use type of vertex weights
        }
        _ => unreachable!(),
    }
}

#[polars_expr(output_type_func=struct_use_weights)]
pub fn premapped_wect(inputs: &[Series], kwargs: PremappedWectArgs) -> PolarsResult<Series> {
    let device = tch::Device::cuda_if_available();
    let wp = WECTParams::new(
        kwargs.embedded_dimension,
        kwargs.num_directions,
        kwargs.num_heights,
        device,
    );

    let psimps = &kwargs.provided_simplices;
    let pweights = &kwargs.provided_weights;

    let vdim = kwargs.embedded_dimension as usize;

    let closure = |complex: &mut WeightedOptComplex<f32, f32>| {
        complex.interpolate_missing_down();
        let pre_rots = complex.premap(
            kwargs.align_dimension,
            kwargs.subsample_ratio,
            kwargs.subsample_min,
            kwargs.subsample_max,
            kwargs.eps,
            kwargs.copies,
        );

        let device = wp.dirs.device();
        let tensor_complex = WeightedTensorComplex::from(&complex, device);
        let wects: Vec<Array2<f32>> = pre_rots
            .iter()
            .map(|x| {
                let tx = array2_to_tensor(x, device);
                let wect = tensor_complex.pre_rot_wect(&wp, tx);
                let wect_arr = tensor_to_array2(
                    &wect,
                    kwargs.num_directions as usize, // i64 to usize conversion
                    kwargs.num_heights as usize,
                );
                wect_arr
            })
            .collect();

        let flattened_wects: Vec<f32> = wects // TODO: unhardcode
            .into_iter()
            .map(|x| x.into_raw_vec())
            .flatten()
            .collect();
        flattened_wects
    };

    iter_complex(&inputs[0], &inputs[1], vdim, psimps, pweights, closure)
}

#[derive(Clone, Deserialize)]
struct WectArgs {
    embedded_dimension: i64,
    num_heights: i64,
    num_directions: i64,
    provided_simplices: Vec<usize>, // the dimensions of simplices provied, in order, starting with 1 // TODO: struct
    provided_weights: Vec<usize>,   // the dimensions of weights provided, in order
}

#[polars_expr(output_type_func=struct_use_weights)]
pub fn wect(inputs: &[Series], kwargs: WectArgs) -> PolarsResult<Series> {
    let device = tch::Device::cuda_if_available();
    let wp = WECTParams::new(
        kwargs.embedded_dimension,
        kwargs.num_directions,
        kwargs.num_heights,
        device,
    );

    let psimps = &kwargs.provided_simplices;
    let pweights = &kwargs.provided_weights;

    let vdim = kwargs.embedded_dimension as usize;

    let closure = |complex: &mut WeightedOptComplex<f32, f32>| {
        complex.interpolate_missing_down();

        let device = wp.dirs.device();
        let tensor_complex = WeightedTensorComplex::from(&complex, device);

        let wects: Vec<Array2<f32>> = vec![tensor_to_array2(
            &tensor_complex.wect(&wp),
            kwargs.num_directions as usize,
            kwargs.num_heights as usize,
        )];

        let flattened_wects: Vec<f32> = wects // TODO: unhardcode
            .into_iter()
            .map(|x| x.into_raw_vec())
            .flatten()
            .collect();
        flattened_wects
    };

    iter_complex(&inputs[0], &inputs[1], vdim, psimps, pweights, closure)
}
