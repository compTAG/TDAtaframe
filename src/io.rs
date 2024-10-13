use std::collections::VecDeque;

use crate::complex_opt::{OptComplex, WeightedOptComplex};
use ndarray::{Array2, ArrayView2};
use num_traits::AsPrimitive;
use polars::datatypes::{
    DataType, Float32Type, Float64Type, Int32Type, Int64Type, UInt16Type, UInt32Type, UInt64Type,
};
use polars::prelude::*;
use polars_arrow::array::{Array, PrimitiveArray};

/// Convert simplex lists to 2D usize arrays
fn build_usize_ndarray(
    simplex_series: &Vec<&ChunkedArray<ListType>>,
    i: usize,
    row: usize,
    dtype: &DataType,
) -> PolarsResult<Array2<usize>> {
    let row = simplex_series[i].get_as_series(row).unwrap();
    match dtype {
        DataType::UInt16 => Ok(row.list()?.to_ndarray::<UInt16Type>()?.mapv(|x| x as usize)),
        DataType::UInt32 => Ok(row.list()?.to_ndarray::<UInt32Type>()?.mapv(|x| x as usize)),
        DataType::UInt64 => Ok(row.list()?.to_ndarray::<UInt64Type>()?.mapv(|x| x as usize)),
        DataType::Int64 => Ok(row.list()?.to_ndarray::<Int64Type>()?.mapv(|x| x as usize)),
        DataType::Int32 => Ok(row.list()?.to_ndarray::<Int32Type>()?.mapv(|x| x as usize)),
        _ => polars_bail!(InvalidOperation:format!(
            "dtype {dtype} not supported for iterating a weighted complex, expected UInt32, UInt64, UInt16, Int64, Int32"
        )),
    }
}

// Define iterators over vertices and simplices.
// The macro to call is iter_vert_simp!. The logic is seen
// in impl_iter_vert_simp, with generic extensions by the other macros
macro_rules! impl_iter_vert_simp {
    ($func_name:ident, $ty:ty, $float_dtype:ty, $data_type:expr) => {
        pub fn $func_name(
            vertices_s: &Series,
            simplices_s: &Series,
            mut clfn: impl FnMut(&ArrayView2<$ty>, &ArrayView2<usize>) -> Vec<$ty>,
        ) -> PolarsResult<Series> {
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
                        .to_ndarray::<$float_dtype>()
                        .unwrap();

                    let simplex_array = s
                        .unwrap()
                        .as_ref()
                        .list()
                        .unwrap()
                        .to_ndarray::<UInt32Type>()
                        .unwrap()
                        .mapv(|x| x as usize);

                    let out = clfn(&vert_array.view(), &simplex_array.view());
                    let prim = Box::new(PrimitiveArray::<$ty>::from_vec(out));
                    Some(prim as Box<dyn Array>)
                })
                .collect_ca_with_dtype("".into(), DataType::List(Box::new($data_type)));
            Ok(out.into_series())
        }
    };
}

impl_iter_vert_simp!(iter_vert_simp_f32, f32, Float32Type, DataType::Float32);
impl_iter_vert_simp!(iter_vert_simp_f64, f64, Float64Type, DataType::Float64);

#[macro_export]
macro_rules! iter_vert_simp {
    ($vertices_s:expr, $simplices_s:expr, |$vertices:ident, $simplices:ident| $closure_body:block) => {{
        let vdtype = $vertices_s.dtype().leaf_dtype();
        // let sdtype = $simplices_s.dtype().leaf_dtype();

        match vdtype {
            DataType::Float32 => {
                let mut closure = |$vertices: &ArrayView2<f32>, $simplices: &ArrayView2<usize>| -> Vec<f32> {
                    $closure_body
                };
                iter_vert_simp_f32($vertices_s, $simplices_s, &mut closure)
            },
            DataType::Float64 => {
                let mut closure = |$vertices: &ArrayView2<f64>, $simplices: &ArrayView2<usize>| -> Vec<f64> {
                    $closure_body
                };
                iter_vert_simp_f64($vertices_s, $simplices_s, &mut closure)
            },
            _ => polars_bail!(ComputeError: "Unsupported data type: {:?}", vdtype),
        }
    }};
}

macro_rules! impl_iter_vert_simp_weight {
    ($func_name:ident, $ty:ty, $float_dtype:ty, $ca_method:ident, $data_type:expr) => {
        pub fn $func_name(
            vertices_s: &Series,
            simplices_s: &Series,
            weights_s: &Series,
            mut clfn: impl FnMut(&ArrayView2<$ty>, &ArrayView2<usize>, &Vec<$ty>) -> Vec<$ty>,
        ) -> PolarsResult<Series> {
            let vertices: &ChunkedArray<ListType> = vertices_s.list()?;
            let simplices: &ChunkedArray<ListType> = simplices_s.list()?;
            let weights: &ChunkedArray<ListType> = weights_s.list()?;
            let out: ChunkedArray<ListType> = vertices
                .amortized_iter()
                .zip(simplices.amortized_iter())
                .zip(weights.amortized_iter())
                .map(|((v, s), w)| -> Option<Box<dyn Array>> {
                    let chunked_weights: &ChunkedArray<$float_dtype> =
                        w.as_ref().unwrap().as_ref().$ca_method().unwrap();

                    let w_vec = chunked_weights.to_vec_null_aware().left().unwrap();

                    let vert_array = v
                        .unwrap()
                        .as_ref()
                        .list()
                        .unwrap()
                        .to_ndarray::<$float_dtype>()
                        .unwrap();
                    let simplex_array = s
                        .unwrap()
                        .as_ref()
                        .list()
                        .unwrap()
                        .to_ndarray::<UInt32Type>()
                        .unwrap()
                        .mapv(|x| x as usize);

                    let out = clfn(&vert_array.view(), &simplex_array.view(), &w_vec);
                    let prim = Box::new(PrimitiveArray::<$ty>::from_vec(out));
                    Some(prim as Box<dyn Array>)
                })
                .collect_ca_with_dtype("".into(), DataType::List(Box::new($data_type)));
            Ok(out.into_series())
        }
    };
}

impl_iter_vert_simp_weight!(
    iter_vert_simp_weight_f32,
    f32,
    Float32Type,
    f32,
    DataType::Float32
);
impl_iter_vert_simp_weight!(
    iter_vert_simp_weight_f64,
    f64,
    Float64Type,
    f64,
    DataType::Float64
);

#[macro_export]
macro_rules! iter_vert_simp_weight {
    ($vertices_s:expr, $simplices_s:expr, $weights_s:expr, |$vertices:ident, $simplices:ident, $weights:ident| $closure_body:block) => {{
        let vdtype = $vertices_s.dtype().leaf_dtype();
        // let sdtype = $simplices_s.dtype().leaf_dtype();
        let wdtype = $weights_s.dtype().leaf_dtype();

        if vdtype != wdtype {
            polars_bail!(InvalidOperation:format!(
                "dtype mismatch between vertices and weights, got {vdtype} and {wdtype}"
            ));
        }

        match vdtype {
            DataType::Float32 => {
                let mut closure = |$vertices: &ArrayView2<f32>, $simplices: &ArrayView2<usize>, $weights: &Vec<f32>| -> Vec<f32> {
                    $closure_body
                };
                iter_vert_simp_weight_f32($vertices_s, $simplices_s, $weights_s, &mut closure)
            },
            DataType::Float64 => {
                let mut closure = |$vertices: &ArrayView2<f64>, $simplices: &ArrayView2<usize>, $weights: &Vec<f64>| -> Vec<f64> {
                    $closure_body
                };
                iter_vert_simp_weight_f64($vertices_s, $simplices_s, $weights_s, &mut closure)
            },
            _ => polars_bail!(ComputeError: "Unsupported data type: {:?}", vdtype),
        }
    }};
}
macro_rules! impl_iter_complex {
    ($func_name:ident, $ty:ty, $float_dtype:ty, $data_type:expr) => {
        pub fn $func_name(
            simplices_s: &Series,
            mut complex_fn: impl FnMut(&mut OptComplex<$ty>) -> Vec<$ty>,
        ) -> PolarsResult<Series> {
            let simplex_fields: Vec<Series> = simplices_s.struct_()?.fields_as_series();
            let simplex_series: Vec<&ChunkedArray<ListType>> =
                simplex_fields.iter().map(|x| x.list().unwrap()).collect();

            // let vdtype = simplex_series[0].dtype().leaf_dtype();
            let sdtype = simplex_series[1].dtype().leaf_dtype();

            let out: ChunkedArray<ListType> = simplex_series[0]
                .amortized_iter()
                .enumerate()
                .map(|(j, v)| -> Option<Box<dyn Array>> {
                    let vertices_array: &ListChunked = v.as_ref().unwrap().as_ref().list().unwrap();
                    let vertices = vertices_array.to_ndarray::<$float_dtype>().unwrap();

                    let simplices: VecDeque<Array2<usize>> = (1..simplex_series.len())
                        .map(|i| -> PolarsResult<Array2<usize>> {
                            build_usize_ndarray(&simplex_series, i, j, &sdtype)
                        })
                        .collect::<PolarsResult<VecDeque<Array2<usize>>>>()
                        .unwrap();

                    let psimps: Vec<usize> = simplices.iter().map(|x| x.shape()[1] - 1).collect();

                    let mut complex = OptComplex::from_provided(vertices, simplices, &psimps);

                    let out = complex_fn(&mut complex);

                    let prim = Box::new(PrimitiveArray::<$ty>::from_vec(out));
                    Some(prim as Box<dyn Array>)
                })
                .collect_ca_with_dtype("".into(), DataType::List(Box::new($data_type)));
            Ok(out.into_series())
        }
    };
}

impl_iter_complex!(iter_complex_f32, f32, Float32Type, DataType::Float32);
impl_iter_complex!(iter_complex_f64, f64, Float64Type, DataType::Float64);

#[macro_export]
macro_rules! iter_complex {
    ($simplices_s:expr, |$complex:ident| $closure_body:block) => {{
        let simplex_fields: Vec<Series> = $simplices_s.struct_()?.fields_as_series();
        let vdtype = simplex_fields[0].dtype().leaf_dtype();

        match vdtype {
            DataType::Float32 => {
                let mut closure = |$complex: &mut OptComplex<f32>| -> Vec<f32> {
                    $closure_body
                };
                iter_complex_f32($simplices_s, &mut closure)
            },
            DataType::Float64 => {
                let mut closure = |$complex: &mut OptComplex<f64>| -> Vec<f64> {
                    $closure_body
                };
                iter_complex_f64($simplices_s, &mut closure)
            },
            _ => polars_bail!(ComputeError: "Unsupported vertex data type: {:?}", vdtype),
        }
    }};
}
// Iterate over a weighted complex.
// The macro to call is iter_weighted_complex!
macro_rules! impl_iter_weighted_complex {
    ($func_name:ident, $ty:ty, $float_dtype:ty, $ca_method:ident, $data_type:expr) => {
        pub fn $func_name(
            simplices_s: &Series,
            weights_s: &Series,
            pweights: Vec<usize>,
            mut complex_fn: impl FnMut(&mut WeightedOptComplex<$ty, $ty>) -> Vec<$ty>,
        ) -> PolarsResult<Series> {
            let simplex_fields: Vec<Series> = simplices_s.struct_()?.fields_as_series();
            let simplex_series: Vec<&ChunkedArray<ListType>> =
                simplex_fields.iter().map(|x| x.list().unwrap()).collect();

            let weight_fields: Vec<Series> = weights_s.struct_()?.fields_as_series();
            let weight_series: Vec<&ChunkedArray<ListType>> =
                weight_fields.iter().map(|x| x.list().unwrap()).collect();

            let vdtype = simplex_series[0].dtype().leaf_dtype();
            let sdtype = simplex_series[1].dtype().leaf_dtype();
            let wdtype = weight_series[0].dtype().leaf_dtype();
            if vdtype != wdtype {
                polars_bail!(InvalidOperation:format!(
                    "dtype mismatch between vertices and weights, got {vdtype} and {wdtype}"
                ));
            }

            let out: ChunkedArray<ListType> = simplex_series[0]
                .amortized_iter()
                .enumerate()
                .map(|(j, v)| -> Option<Box<dyn Array>> {
                    let simplices: VecDeque<Array2<usize>> = (1..simplex_series.len())
                        .map(|i| -> PolarsResult<Array2<usize>> {
                            build_usize_ndarray(&simplex_series, i, j, &sdtype)
                        })
                        .collect::<PolarsResult<VecDeque<Array2<usize>>>>()
                        .unwrap();

                    let psimps: Vec<usize> =
                        simplices.iter().map(|x| x.shape()[1] - 1).collect();

                    let vertices_array: &ListChunked =
                        v.as_ref().unwrap().as_ref().list().unwrap();

                    let vertices = vertices_array.to_ndarray::<$float_dtype>().unwrap();

                    let weights: VecDeque<Vec<$ty>> = weight_series
                        .iter()
                        .map(|x| {
                            x.get_as_series(j)
                                .unwrap()
                                .$ca_method()
                                .unwrap()
                                .to_vec_null_aware()
                                .left()
                                .unwrap()
                        })
                        .collect();

                    let mut complex = WeightedOptComplex::from_provided(
                        vertices,
                        simplices,
                        weights,
                        &psimps,
                        &pweights,
                    );

                    let out = complex_fn(&mut complex);

                    let prim = Box::new(PrimitiveArray::<$ty>::from_vec(out));
                    Some(prim as Box<dyn Array>)
                })
                .collect_ca_with_dtype("".into(), DataType::List(Box::new($data_type)));
            Ok(out.into_series())
        }
    };
}

impl_iter_weighted_complex!(
    iter_weighted_complex_f32,
    f32,
    Float32Type,
    f32,
    DataType::Float32
);

impl_iter_weighted_complex!(
    iter_weighted_complex_f64,
    f64,
    Float64Type,
    f64,
    DataType::Float64
);

#[macro_export]
macro_rules! iter_weighted_complex {
    ($simplices_s:expr, $weights_s:expr, $pweights:expr, |$complex:ident| $closure_body:block) => {{
        let simplex_fields: Vec<Series> = $simplices_s.struct_()?.fields_as_series();
        let weight_fields: Vec<Series> = $weights_s.struct_()?.fields_as_series();
        let vdtype = simplex_fields[0].dtype().leaf_dtype();
        let wdtype = weight_fields[0].dtype().leaf_dtype();

        if vdtype != wdtype {
            polars_bail!(InvalidOperation:format!(
                "dtype mismatch between vertices and weights, got {vdtype} and {wdtype}"
            ));
        }

        match vdtype {
            DataType::Float32 => {
                let mut closure = |$complex: &mut WeightedOptComplex<f32, f32>| -> Vec<f32> {
                    $closure_body
                };
                iter_weighted_complex_f32($simplices_s, $weights_s, $pweights, &mut closure)
            },
            DataType::Float64 => {
                let mut closure = |$complex: &mut WeightedOptComplex<f64, f64>| -> Vec<f64> {
                    $closure_body
                };
                iter_weighted_complex_f64($simplices_s, $weights_s, $pweights, &mut closure)
            },
            _ => polars_bail!(ComputeError: "Unsupported data type: {:?}", vdtype),
        }
    }};
}
