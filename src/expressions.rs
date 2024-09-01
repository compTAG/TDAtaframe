use std::collections::VecDeque;

use crate::complex::{WeightedOptComplex, WeightedTensorComplex};
use crate::complex_interpolation::Interpolate;
use crate::complex_mapping::PreMappable;
use crate::complex_mapping::{compute_barycenters, compute_maps_svd};
use crate::tensorwect::{TensorWect, WECTParams};
use crate::utils::{array2_to_tensor, tensor_to_array2};
use itertools::Itertools;
use ndarray::{Array2, ArrayView2};
use polars_core::utils::arrow::array::{Array, PrimitiveArray};
use polars_core::utils::rayon::iter::{IndexedParallelIterator, IntoParallelIterator, MultiZip};
use polars_core::utils::rayon::prelude::*;
use serde::Deserialize;

use pyo3_polars::derive::polars_expr;

use pyo3_polars::export::polars_core::prelude::*;

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

            let maps: Vec<Array2<f32>> = compute_maps_svd(
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

fn impl_pmw3d(
    chunked_vertices: &ChunkedArray<Float32Type>,
    chunked_simplices: &ChunkedArray<UInt32Type>,
    chunked_normals: &ChunkedArray<Float32Type>,
    kwargs: &PremappedWect3Args,
    wp: &WECTParams,
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

            let mut complex = WeightedOptComplex::from_simplices(
                vert_array.into_owned(),
                vec![None, Some(simplex_array.into_owned())],
                vec![None, None, Some(normals)],
            );

            complex.interpolate_missing_down();
            let pre_rots = complex.premap(
                2,
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
                    let wect = tensor_complex.pre_rot_wect(wp, tx);
                    let wect_arr = tensor_to_array2(
                        &wect,
                        kwargs.num_directions as usize, // i64 to usize conversion
                        kwargs.num_heights as usize,
                    );
                    wect_arr
                })
                .collect();

            // let wects: Vec<Array2<f32>> = vec![tensor_to_array2(
            //     &tensor_complex.wect(&wp),
            //     kwargs.num_directions as usize,
            //     kwargs.num_heights as usize,
            // )];

            let flattened_wects: Vec<f32> = wects // TODO: unhardcode
                .into_iter()
                .map(|x| x.into_raw_vec())
                .flatten()
                .collect();

            // TODO: remove generic
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(flattened_wects)); // TODO: return
            Some(prim)
        }
        _ => panic!("Expected exactly least one of the two to be Some"),
    }
}

#[derive(Clone, Deserialize)]
struct PremappedWectArgs {
    embedded_dimension: i64,
    num_heights: i64,
    num_directions: i64,
    provided_simplices: Vec<usize>, // the dimensions of simplices provied, in order, starting with 1 // TODO: struct
    provided_weights: Vec<usize>,   // the dimensions of weights provided, in order
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f32>,
    copies: bool,
}

#[derive(Clone, Copy, Deserialize)]
struct PremappedWect3Args {
    embedded_dimension: i64,
    num_heights: i64,
    num_directions: i64,
    // provided_simplices: Vec<usize>, // the dimensions of simplices provied, in order // TODO: struct
    // provided_weights: Vec<usize>,   // the dimensions of weights provided, in order
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f32>,
    copies: bool,
}

#[polars_expr(output_type_func=same_output_type)] // TODO: write better output type function
pub fn premapped_wect3(inputs: &[Series], kwargs: PremappedWect3Args) -> PolarsResult<Series> {
    // let n_simp = kwargs.provided_simplices.len();
    // let n_weight = kwargs.provided_weights.len();

    // maybe make this Opt<chunked>
    // let chunks: Vec<&ChunkedArray<ListType>> = inputs
    //     .iter()
    //     .map(|x| x.list().unwrap()) // TODO: get the ? in there
    //     .collect();
    let device = tch::Device::cuda_if_available();
    let wp = WECTParams::new(
        kwargs.embedded_dimension,
        kwargs.num_directions,
        kwargs.num_heights,
        device,
    );

    let vertices: &ChunkedArray<ListType> = inputs[0].list()?;
    let triangles: &ChunkedArray<ListType> = inputs[1].list()?;
    let weights: &ChunkedArray<ListType> = inputs[2].list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter()
        .zip(triangles.amortized_iter())
        .zip(weights.amortized_iter())
        .map(|((v, s), n)| {
            let chunked_vertices: &ChunkedArray<Float32Type> = v.as_ref()?.as_ref().f32().unwrap();
            let chunked_simplices: &ChunkedArray<UInt32Type> = s.as_ref()?.as_ref().u32().unwrap();
            let chunked_normals: &ChunkedArray<Float32Type> = n.as_ref()?.as_ref().f32().unwrap();
            impl_pmw3d(
                &chunked_vertices,
                &chunked_simplices,
                &chunked_normals,
                &kwargs,
                &wp,
            )
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));

    // call impl here?
    Ok(out.into_series())
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
    if psimps.len() == 0 && psimps[0] == 0 {
        panic!("Provided simplices must be greater than 0");
    }
    let pweights = &kwargs.provided_weights;

    let k = psimps.get(psimps.len() - 1).unwrap();
    let vdim = kwargs.embedded_dimension as usize;

    let simplex_fields: Vec<Series> = inputs[0].struct_()?.fields_as_series();
    let simplex_series: Vec<&ChunkedArray<ListType>> =
        simplex_fields.iter().map(|x| x.list().unwrap()).collect();
    let weight_fields: Vec<Series> = inputs[1].struct_()?.fields_as_series();
    let weight_series: Vec<&ChunkedArray<ListType>> =
        weight_fields.iter().map(|x| x.list().unwrap()).collect();

    let out: ChunkedArray<ListType> = simplex_series[0]
        .amortized_iter() // TODO: maybe want to get rid of amortized
        .enumerate()
        .map(|(j, v)| -> Option<Box<dyn Array>> {
            // j is the index of the row
            let vertices: Vec<f32> = v
                .as_ref()
                .unwrap()
                .as_ref()
                .f32()
                .unwrap()
                .to_vec_null_aware()
                .left()
                .unwrap();
            // i is the index into the Series vector
            let mut simplices: VecDeque<Vec<usize>> = (1..simplex_series.len())
                .map(|i| -> Vec<usize> {
                    simplex_series[i]
                        .get_as_series(j)
                        .unwrap()
                        .u32()
                        .unwrap()
                        .to_vec_null_aware()
                        .left()
                        .unwrap()
                        .into_iter()
                        .map(|x| x as usize)
                        .collect::<Vec<usize>>()
                })
                .collect();

            let mut weights: VecDeque<Vec<f32>> = (0..weight_series.len())
                .map(|i| -> Vec<f32> {
                    weight_series[i]
                        .get_as_series(j)
                        .unwrap()
                        .f32()
                        .unwrap()
                        .to_vec_null_aware()
                        .left()
                        .unwrap()
                })
                .collect();

            let vertices: Array2<f32> =
                ArrayView2::from_shape((vertices.len() / vdim, vdim), &vertices)
                    .unwrap()
                    .into_owned(); // TODO: remove into_owned. this needs modification in
                                   // complex.rs
            let mut opt_simplices: Vec<Option<Array2<usize>>> = Vec::with_capacity(*k);
            let mut opt_weights: Vec<Option<Vec<f32>>> = Vec::with_capacity(*k + 1);

            let mut i = 1; // i tracks the current dimension
            psimps.iter().for_each(|&dim| {
                while i < dim {
                    // fill missing dimensions
                    opt_simplices.push(None);
                    i += 1;
                }

                let s_row = simplices.pop_front().unwrap();
                let simplex_array =
                    ArrayView2::from_shape((s_row.len() / (dim + 1), (dim + 1)), &s_row).unwrap(); // TODO: no
                                                                                                   // copy
                opt_simplices.push(Some(simplex_array.into_owned()));
                i += 1;
            });

            let mut i = 0;
            pweights.iter().for_each(|&dim| {
                while i < dim {
                    opt_weights.push(None);
                    i += 1;
                }
                opt_weights.push(Some(weights.pop_front().unwrap()));
                i += 1;
            });

            let mut complex =
                WeightedOptComplex::from_simplices(vertices, opt_simplices, opt_weights);

            complex.interpolate_missing_down();
            let pre_rots = complex.premap(
                2,
                kwargs.subsample_ratio,
                kwargs.subsample_min,
                kwargs.subsample_max,
                kwargs.eps,
                kwargs.copies,
            );

            let device = wp.dirs.device();
            let tensor_complex = WeightedTensorComplex::from(&complex, device);
            // let wects: Vec<Array2<f32>> = pre_rots
            //     .iter()
            //     .map(|x| {
            //         let tx = array2_to_tensor(x, device);
            //         let wect = tensor_complex.pre_rot_wect(wp, tx);
            //         let wect_arr = tensor_to_array2(
            //             &wect,
            //             kwargs.num_directions as usize, // i64 to usize conversion
            //             kwargs.num_heights as usize,
            //         );
            //         wect_arr
            //     })
            //     .collect();

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

            // TODO: remove generic
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(flattened_wects));
            Some(prim)
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));
    Ok(out.into_series())
}
