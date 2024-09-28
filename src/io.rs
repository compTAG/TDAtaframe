use std::collections::VecDeque;

use crate::complex_opt::{OptComplex, WeightedOptComplex};
use ndarray::{Array2, ArrayView2};
// use polars_core::utils::rayon::iter::{IndexedParallelIterator, IntoParallelIterator, MultiZip};
// use polars_core::utils::rayon::prelude::*;
use polars::datatypes::ListChunked;
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};
// use pyo3_polars::export::polars_core::prelude::*;

pub fn iter_vert_simp_weight(
    vertices_s: &Series,
    simplices_s: &Series,
    weights_s: &Series,

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
            let chunked_weights: &ChunkedArray<Float32Type> =
                w.as_ref().unwrap().as_ref().f32().unwrap();

            let mut w_vec = chunked_weights.to_vec_null_aware().left().unwrap();

            let vert_array = v
                .unwrap()
                .as_ref()
                .list()
                .unwrap()
                .to_ndarray::<Float32Type>()
                .unwrap();
            let simplex_array = s
                .unwrap()
                .as_ref()
                .list()
                .unwrap()
                .to_ndarray::<UInt32Type>()
                .unwrap()
                .mapv(|x| x as usize);

            let out = clfn(
                &mut vert_array.view(),
                &mut simplex_array.view(),
                &mut w_vec,
            );
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("".into(), DataType::List(Box::new(DataType::Float32)));

    Ok(out.into_series())
}

pub fn iter_vert_simp(
    vertices_s: &Series,
    simplices_s: &Series,
    mut clfn: impl FnMut(&mut ArrayView2<f32>, &mut ArrayView2<usize>) -> Vec<f32>,
) -> PolarsResult<Series> {
    // TODO: valid input checks
    let vertices: &ChunkedArray<ListType> = vertices_s.list()?;
    let simplices: &ChunkedArray<ListType> = simplices_s.list()?;
    let out: ChunkedArray<ListType> = vertices
        .amortized_iter()
        .zip(simplices.amortized_iter())
        .map(|(v, s)| -> Option<Box<dyn Array>> {
            let vert_array = v
                .unwrap()
                .as_ref()
                .list()
                .unwrap()
                .to_ndarray::<Float32Type>()
                .unwrap();

            let simplex_array = s
                .unwrap()
                .as_ref()
                .list()
                .unwrap()
                .to_ndarray::<UInt32Type>()
                .unwrap()
                .mapv(|x| x as usize);

            let out = clfn(&mut vert_array.view(), &mut simplex_array.view());
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("".into(), DataType::List(Box::new(DataType::Float32)));

    Ok(out.into_series())
}

pub fn iter_complex(
    simplices_s: &Series,
    mut complex_fn: impl FnMut(&mut OptComplex<f32>) -> Vec<f32>,
) -> PolarsResult<Series> {
    // get each field in struct as a list
    let simplex_fields: Vec<Series> = simplices_s.struct_()?.fields_as_series();
    let simplex_series: Vec<&ChunkedArray<ListType>> =
        simplex_fields.iter().map(|x| x.list().unwrap()).collect();

    let out: ChunkedArray<ListType> = simplex_series[0]
        .amortized_iter() // HACK: maybe want to get rid of amortized
        .enumerate()
        .map(|(j, v)| -> Option<Box<dyn Array>> {
            // j is the index of the row
            let vertices_array: &ListChunked = v.as_ref().unwrap().as_ref().list().unwrap();
            let vertices = vertices_array.to_ndarray::<Float32Type>().unwrap();

            // i is the index into the Series vector
            let mut simplices: VecDeque<Array2<usize>> = (1..simplex_series.len())
                .map(|i| -> Array2<usize> {
                    simplex_series[i]
                        .get_as_series(j)
                        .unwrap()
                        .list()
                        .unwrap()
                        .to_ndarray::<UInt32Type>()
                        .unwrap()
                        .mapv(|x| x as usize)
                })
                .collect();

            // instead, build psimps by looking at the sizes of the arrays in simplices
            let psimps: Vec<usize> = simplices.iter().map(|x| x.shape()[1] - 1).collect();

            if psimps.len() == 0 && psimps[0] == 0 {
                panic!("Provided simplices must be greater than 0");
            }
            let k = psimps.get(psimps.len() - 1).unwrap(); // the top dimension

            let mut opt_simplices: Vec<Option<Array2<usize>>> = Vec::with_capacity(*k);

            let mut i = 1; // i tracks the current dimension
            psimps.iter().for_each(|&dim| {
                while i < dim {
                    // fill missing dimensions
                    opt_simplices.push(None);
                    i += 1;
                }

                let simplex_array = simplices.pop_front().unwrap();
                opt_simplices.push(Some(simplex_array.into_owned()));
                i += 1;
            });

            let mut complex = OptComplex::from_simplices(vertices, opt_simplices);

            let out = complex_fn(&mut complex);

            // TODO: make generic
            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("".into(), DataType::List(Box::new(DataType::Float32)));
    Ok(out.into_series())
}

pub fn iter_weighted_complex(
    simplices_s: &Series,
    weights_s: &Series,
    mut complex_fn: impl FnMut(&mut WeightedOptComplex<f32, f32>) -> Vec<f32>,
) -> PolarsResult<Series> {
    // get each field in struct as a list
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
            let vertices_array: &ListChunked = v.as_ref().unwrap().as_ref().list().unwrap();
            let vertices = vertices_array.to_ndarray::<Float32Type>().unwrap();

            // i is the index into the Series vector
            let mut simplices: VecDeque<Array2<usize>> = (1..simplex_series.len())
                .map(|i| -> Array2<usize> {
                    simplex_series[i]
                        .get_as_series(j)
                        .unwrap()
                        .list()
                        .unwrap()
                        .to_ndarray::<UInt32Type>()
                        .unwrap()
                        .mapv(|x| x as usize)
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

            // instead, build psimps by looking at the sizes of the arrays in simplices
            let psimps: Vec<usize> = simplices.iter().map(|x| x.shape()[1] - 1).collect();
            let pweights: Vec<usize> = weights.iter().map(|x| x.len()).collect();

            if psimps.len() == 0 && psimps[0] == 0 {
                panic!("Provided simplices must be greater than 0");
            }
            let k = psimps.get(psimps.len() - 1).unwrap(); // the top dimension

            let mut opt_simplices: Vec<Option<Array2<usize>>> = Vec::with_capacity(*k);
            let mut opt_weights: Vec<Option<Vec<f32>>> = Vec::with_capacity(k + 1);

            let mut i = 1; // i tracks the current dimension
            psimps.iter().for_each(|&dim| {
                while i < dim {
                    // fill missing dimensions
                    opt_simplices.push(None);
                    i += 1;
                }

                let simplex_array = simplices.pop_front().unwrap();
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

            let prim = Box::new(PrimitiveArray::<f32>::from_vec(out));
            Some(prim)
        })
        .collect_ca_with_dtype("".into(), DataType::List(Box::new(DataType::Float32)));
    Ok(out.into_series())
}
