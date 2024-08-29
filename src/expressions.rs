use crate::complex_mapping::{compute_barycenters, compute_maps_svd};
use ndarray::{Array2, ArrayView2};
use polars::error::PolarsResult;
use serde::Deserialize;

use polars_arrow::array::{Array, PrimitiveArray};

use pyo3_polars::derive::polars_expr;

use polars::prelude::*;

fn same_output_type(input_fields: &[Field]) -> PolarsResult<Field> {
    let field = &input_fields[0];
    Ok(field.clone())
}

fn impl_compute_barycenters(
    chunked_vertices: &ChunkedArray<Float32Type>,
    chunked_simplices: &ChunkedArray<UInt32Type>,
) -> Option<Box<dyn Array>> {
    const DIMV: usize = 3; // TODO: make this a parameter
    const DIMS: usize = 3; // TODO: make this a parameter

    let v_flat = chunked_vertices.to_vec_null_aware().left();
    let s_flat = chunked_simplices.to_vec_null_aware().left();

    match (v_flat, s_flat) {
        (Some(vertices), Some(simplices)) => {
            let simplices = simplices
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>();
            let vert_array =
                ArrayView2::from_shape((chunked_vertices.len() / DIMV, DIMV), &vertices).unwrap();
            let simplex_array =
                ArrayView2::from_shape((chunked_simplices.len() / DIMS, DIMS), &simplices).unwrap();

            let barycenters = compute_barycenters(&vert_array, &simplex_array).into_raw_vec();

            let prim = Box::new(PrimitiveArray::<f32>::from_vec(barycenters));
            Some(prim)
        }
        _ => panic!("Expected exactly least one of the two to be Some"),
    }
}

#[polars_expr(output_type_func=same_output_type)] // TODO: Make generic
pub fn barycenters(inputs: &[Series]) -> PolarsResult<Series> {
    let vertices: &ChunkedArray<ListType> = inputs[0].list()?;
    let simplices: &ChunkedArray<ListType> = inputs[1].list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter()
        .zip(simplices.amortized_iter())
        .map(|(v, s)| {
            let chunked_vertices: &ChunkedArray<Float32Type> = v.as_ref()?.as_ref().f32().unwrap();
            let chunked_simplices: &ChunkedArray<UInt32Type> = s.as_ref()?.as_ref().u32().unwrap();
            impl_compute_barycenters(&chunked_vertices, &chunked_simplices)
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));

    // call impl here?
    Ok(out.into_series())
}

#[derive(Clone, Copy, Deserialize)]
struct MapsSvdArgs {
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f32>,
    copies: bool,
}

fn impl_maps_svd(
    chunked_vertices: &ChunkedArray<Float32Type>,
    chunked_simplices: &ChunkedArray<UInt32Type>,
    chunked_normals: &ChunkedArray<Float32Type>,
    kwargs: &MapsSvdArgs,
) -> Option<Box<dyn Array>> {
    const DIMV: usize = 3; // TODO: make this a parameter
    const DIMS: usize = 3; // TODO: make this a parameter

    let v_flat = chunked_vertices.to_vec_null_aware().left();
    let s_flat = chunked_simplices.to_vec_null_aware().left();
    let normals = chunked_normals.to_vec_null_aware().left();

    match (v_flat, s_flat, normals) {
        (Some(vertices), Some(simplices), Some(normals)) => {
            let simplices = simplices
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>();
            let vert_array =
                ArrayView2::from_shape((chunked_vertices.len() / DIMV, DIMV), &vertices).unwrap();
            let simplex_array =
                ArrayView2::from_shape((chunked_simplices.len() / DIMS, DIMS), &simplices).unwrap();
            let normal_array =
                ArrayView2::from_shape((chunked_normals.len() / DIMV, DIMV), &normals).unwrap();

            let mut maps: Vec<Array2<f32>> = compute_maps_svd(
                // TODO: unhardcode f32
                // TODO: get these from faer
                &vert_array,
                &simplex_array,
                &normals,
                kwargs.subsample_ratio,
                kwargs.subsample_min,
                kwargs.subsample_max,
                kwargs.eps,
                kwargs.copies,
            );

            let flattened_maps: Vec<f32> = maps // TODO: unhardcode
                .into_iter()
                .map(|x| x.into_raw_vec())
                .flatten()
                .collect();

            // TODO: remove generic
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(flattened_maps)); // TODO: return
                                                                                  // as unflattened
            Some(prim)
        }
        _ => panic!("Expected exactly least one of the two to be Some"),
    }
}

#[polars_expr(output_type_func=same_output_type)] // TODO: when unflattened, need to change output
pub fn maps_svd(inputs: &[Series], kwargs: MapsSvdArgs) -> PolarsResult<Series> {
    let vertices: &ChunkedArray<ListType> = inputs[0].list()?;
    let simplices: &ChunkedArray<ListType> = inputs[1].list()?;
    let normals: &ChunkedArray<ListType> = inputs[2].list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter()
        .zip(simplices.amortized_iter())
        .zip(normals.amortized_iter()) // HACK: triple zip better way?
        .map(|((v, s), n)| {
            let chunked_vertices: &ChunkedArray<Float32Type> = v.as_ref()?.as_ref().f32().unwrap();
            let chunked_simplices: &ChunkedArray<UInt32Type> = s.as_ref()?.as_ref().u32().unwrap();
            let chunked_normals: &ChunkedArray<Float32Type> = n.as_ref()?.as_ref().f32().unwrap();
            impl_maps_svd(
                &chunked_vertices,
                &chunked_simplices,
                &chunked_normals,
                &kwargs,
            )
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));

    // call impl here?
    Ok(out.into_series())
}

struct PremappedWectArgs {
    embedded_dimension: i64,
    num_heights: i64,
    num_directions: i64,
    provided_simplices: Vec<usize>, // the dimensions of simplices provied, in order // TODO: struct
    provided_weights: Vec<usize>,   // the dimensions of weights provided, in order
}
pub fn premapped_wect(inputs: &[Series], kwargs: PremappedWectArgs) -> PolarsResult<Series> {
    let n_simp = kwargs.provided_simplices.len();
    let n_weight = kwargs.provided_weights.len();

    // maybe make this Opt<chunked>
    let simplices: Vec<&ChunkedArray<ListType>> = inputs[0..n_simp]
        .iter()
        .map(|x| x.list().unwrap()) // TODO: get the ? in there
        .collect();

    let weights: Vec<&ChunkedArray<ListType>> = inputs[n_simp..n_weight]
        .iter()
        .map(|x| x.list().unwrap())
        .collect();

    // TODO build iterator from each element of simplices and each element of weights
}
