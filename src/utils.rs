use ndarray::{Array2, Axis};
use polars::prelude::*;
use tch;

// This function is useful for writing functions which
// accept pairs of List columns. Delete if unneded.
#[allow(dead_code)]
pub(crate) fn binary_amortized_elementwise<'a, T, K, F>(
    ca: &'a ListChunked,
    weights: &'a ListChunked,
    mut f: F,
) -> ChunkedArray<T>
where
    T: PolarsDataType,
    T::Array: ArrayFromIter<Option<K>>,
    F: FnMut(&Series, &Series) -> Option<K> + Copy,
{
    {
        ca.amortized_iter()
            .zip(weights.amortized_iter())
            .map(|(lhs, rhs)| match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => f(lhs.as_ref(), rhs.as_ref()),
                _ => None,
            })
            .collect_ca(ca.name())
    }
}

pub fn array2_to_tensor<E: tch::kind::Element>(
    array: &Array2<E>,
    device: tch::Device,
) -> tch::Tensor {
    let ax0 = array.len_of(Axis(0));
    let ax1 = array.len_of(Axis(1));
    let tensor = tch::Tensor::from_slice(array.as_slice().unwrap())
        .view([ax0 as i64, ax1 as i64])
        .to_device(device);
    tensor
}

pub fn tensor_to_array2<O: tch::kind::Element + Default>(
    tensor: &tch::Tensor,
    length: usize,
    width: usize,
) -> Array2<O> {
    let mut dst = vec![O::default(); length * width];
    tensor
        .to_kind(tch::Kind::Float)
        .to_device(tch::Device::Cpu)
        .copy_data(&mut dst, length * width);
    Array2::from_shape_vec((length, width), dst).unwrap()
}

pub fn tensor_to_flat<O: tch::kind::Element + Default>(tensor: &tch::Tensor) -> Vec<O> {
    let num_elems = tensor.size().into_iter().fold(1, |acc, x| acc * x) as usize;
    let mut dst = vec![O::default(); num_elems];
    tensor
        .to_kind(tch::Kind::Float)
        .to_device(tch::Device::Cpu)
        .copy_data(&mut dst, num_elems);
    dst
}
