use crate::complex::{WeightedArrayComplex, WeightedTensorComplex};
use crate::utils::tensor_to_array2;
use ndarray::Array2;
use tch;

const DIRECTIONS: usize = 100;
const STEPS: usize = 100;

fn wect_weighted_triangles(
    vertices: Array2<f32>,
    triangles: Array2<usize>,
    face_normals: Vec<f32>,
    device: tch::Device,
) -> Result<Array2<f32>, tch::TchError> {
    let mut complex = WeightedArrayComplex::from_simplices(
        vertices,
        vec![None, Some(triangles)],
        vec![None, None, Some(face_normals)],
    );
    complex.interpolate_missing_down();

    let tensor_complex = WeightedTensorComplex::from(complex, device);

    let wect_tensor = wect(tensor_complex)?;

    Ok(tensor_to_array2(&wect_tensor, DIRECTIONS, STEPS))
}

// FIXME: recompile module from python if proper dir/steps not found
fn wect(w: WeightedTensorComplex) -> Result<tch::Tensor, tch::TchError> {
    // load torchscript model
    let mut model = tch::jit::CModule::load("wectmod.pt")?;
    model.set_eval();
    let wect_tensor = model.forward_is(&w.zip_into_ival())?;
    Ok(wect_tensor.try_into()?)
}
