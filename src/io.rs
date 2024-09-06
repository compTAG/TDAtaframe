use std::collections::VecDeque;

use crate::complex::WeightedOptComplex;
use ndarray::{Array2, ArrayView2};
// use polars_core::utils::rayon::iter::{IndexedParallelIterator, IntoParallelIterator, MultiZip};
// use polars_core::utils::rayon::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
use pyo3_polars::export::polars_core::prelude::*;

pub fn iter_vert_simp_weight(
    vertices_s: &Series,
    simplices_s: &Series,
    weights_s: &Series,
    vdim: usize,
    sdim: usize,

    mut clfn: impl FnMut(&mut ArrayView2<f32>, &mut ArrayView2<usize>, &mut Vec<f32>) -> Vec<f32>,
) -> PolarsResult<Series> {
    // TODO: valid input checks
    let vertices: &ChunkedArray<ListType> = vertices_s.list()?;
    let simplices: &ChunkedArray<ListType> = simplices_s.list()?;
    let weights: &ChunkedArray<ListType> = weights_s.list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter() // HACK: Is amortized iter correct here?
        .zip(simplices.amortized_iter())
        .zip(weights.amortized_iter())
        .map(|((v, s), w)| -> Option<Box<dyn Array>> {
            let chunked_vertices: &ChunkedArray<Float32Type> =
                v.as_ref().unwrap().as_ref().f32().unwrap();
            let chunked_simplices: &ChunkedArray<UInt32Type> =
                s.as_ref().unwrap().as_ref().u32().unwrap();
            let chunked_weights: &ChunkedArray<Float32Type> =
                w.as_ref().unwrap().as_ref().f32().unwrap();

            let v_vec = chunked_vertices.to_vec_null_aware().left().unwrap();
            let s_vec = chunked_simplices
                .to_vec_null_aware()
                .left()
                .unwrap()
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>();

            let mut w_vec = chunked_weights.to_vec_null_aware().left().unwrap();

            let mut vert_array =
                ArrayView2::from_shape((v_vec.len() / vdim, vdim), &v_vec).unwrap();
            let mut simplex_array =
                ArrayView2::from_shape((s_vec.len() / (sdim + 1), (sdim + 1)), &s_vec).unwrap();

            let out = clfn(&mut vert_array, &mut simplex_array, &mut w_vec);
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));

    Ok(out.into_series())
}

pub fn iter_vert_simp(
    vertices_s: &Series,
    simplices_s: &Series,
    vdim: usize,
    sdim: usize,
    mut clfn: impl FnMut(&mut ArrayView2<f32>, &mut ArrayView2<usize>) -> Vec<f32>,
) -> PolarsResult<Series> {
    // TODO: valid input checks
    let vertices: &ChunkedArray<ListType> = vertices_s.list()?;
    let simplices: &ChunkedArray<ListType> = simplices_s.list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter()
        .zip(simplices.amortized_iter())
        .map(|(v, s)| -> Option<Box<dyn Array>> {
            let chunked_vertices: &ChunkedArray<Float32Type> =
                v.as_ref().unwrap().as_ref().f32().unwrap();
            let chunked_simplices: &ChunkedArray<UInt32Type> =
                s.as_ref().unwrap().as_ref().u32().unwrap();

            let v_vec = chunked_vertices.to_vec_null_aware().left().unwrap();
            let s_vec = chunked_simplices
                .to_vec_null_aware()
                .left()
                .unwrap()
                .into_iter()
                .map(|x| x as usize)
                .collect::<Vec<usize>>();

            let mut vert_array =
                ArrayView2::from_shape((v_vec.len() / vdim, vdim), &v_vec).unwrap();
            let mut simplex_array =
                ArrayView2::from_shape((s_vec.len() / (sdim + 1), (sdim + 1)), &s_vec).unwrap();

            let out = clfn(&mut vert_array, &mut simplex_array);
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));

    Ok(out.into_series())
}

pub fn iter_complex(
    simplices_s: &Series,
    weights_s: &Series,
    vdim: usize,
    psimps: &Vec<usize>,   //provided simplices, sorted, 1 to k
    pweights: &Vec<usize>, //provided weights, sorted, 0 to k
    mut complex_fn: impl FnMut(&mut WeightedOptComplex<f32, f32>) -> Vec<f32>,
) -> PolarsResult<Series> {
    if psimps.len() == 0 && psimps[0] == 0 {
        panic!("Provided simplices must be greater than 0");
    }

    let k = psimps.get(psimps.len() - 1).unwrap();

    let simplex_fields: Vec<Series> = simplices_s.struct_()?.fields_as_series();
    let simplex_series: Vec<&ChunkedArray<ListType>> =
        simplex_fields.iter().map(|x| x.list().unwrap()).collect();
    let weight_fields: Vec<Series> = weights_s.struct_()?.fields_as_series();
    let weight_series: Vec<&ChunkedArray<ListType>> =
        weight_fields.iter().map(|x| x.list().unwrap()).collect();

    let out: ChunkedArray<ListType> = simplex_series[0]
        .amortized_iter() // HACK: maybe want to get rid of amortized
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
            let mut opt_weights: Vec<Option<Vec<f32>>> = Vec::with_capacity(k + 1);

            let mut i = 1; // i tracks the current dimension
            psimps.iter().for_each(|&dim| {
                while i < dim {
                    // fill missing dimensions
                    opt_simplices.push(None);
                    i += 1;
                }

                let s_row = simplices.pop_front().unwrap();
                let simplex_array =
                    ArrayView2::from_shape((s_row.len() / (dim + 1), (dim + 1)), &s_row).unwrap(); // TODO: no copy
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

            let out = complex_fn(&mut complex);

            // TODO: make generic
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("", DataType::List(Box::new(DataType::Float32)));
    Ok(out.into_series())
}
