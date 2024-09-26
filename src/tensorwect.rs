use crate::{
    complex::{Complex, Weighted},
    complex_opt::{OptComplex, WeightedOptComplex},
    complex_tensor::{TensorComplex, WeightedTensorComplex},
    utils::array2_to_tensor,
};
use ndarray::Array2;
use tch::{kind::Element, Device, Kind, Tensor};

#[derive(Debug)]
pub struct ECTParams {
    pub dirs: Tensor,
    pub num_heights: i64,
}

impl ECTParams {
    pub fn from_dirs(dirs: Tensor, num_heights: i64) -> Self {
        // let device = dirs.device();
        // // let height_tensor = Tensor::scalar_tensor(num_heights, (kind::Kind::Int64, device));
        ECTParams {
            dirs: dirs.set_requires_grad(false),
            num_heights, //height_tensor.set_requires_grad(false),
        }
    }

    pub fn new(embedded_dimension: i64, num_dirs: i64, num_heghts: i64, device: Device) -> Self {
        let dirs = sample_dirs(num_dirs, embedded_dimension, device);
        Self::from_dirs(dirs, num_heghts)
    }
}

pub trait TensorEct {
    type RotMat;
    // Computes the ECT for the complex, applying a rotation matrix to the vertices beforehand.
    fn pre_rot_ect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor;

    // Computes the ECT for the complex.
    fn ect(&self, params: &ECTParams) -> Tensor;
}

pub trait TensorWect {
    type RotMat;
    // Computes the WECT for the complex, applying a rotation matrix to the vertices beforehand.
    fn pre_rot_wect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor;

    // Computes the WECT for the complex.
    fn wect(&self, params: &ECTParams) -> Tensor;
}

impl TensorWect for WeightedTensorComplex {
    type RotMat = Tensor;
    fn wect(&self, params: &ECTParams) -> Tensor {
        wect(self, None, params)
    }

    fn pre_rot_wect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        wect(self, Some(tx), params)
    }
}

impl TensorEct for WeightedTensorComplex {
    type RotMat = Tensor;
    fn ect(&self, params: &ECTParams) -> Tensor {
        ect(&self.structure, None, params)
    }

    fn pre_rot_ect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        ect(&self.structure, Some(tx), params)
    }
}

impl TensorEct for TensorComplex {
    type RotMat = Tensor;
    fn ect(&self, params: &ECTParams) -> Tensor {
        ect(self, None, params)
    }

    fn pre_rot_ect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        ect(self, Some(tx), params)
    }
}

impl<P, W> TensorWect for WeightedOptComplex<P, W>
where
    P: Element,
    W: Element,
{
    type RotMat = Array2<f64>;
    fn wect(&self, params: &ECTParams) -> Tensor {
        if self.has_missing_dims() {
            panic!("Cannot compute WECT with missing dimensions");
        }
        let device = tch::Device::cuda_if_available();
        let tensor_complex = WeightedTensorComplex::from(self, device);
        tensor_complex.wect(params)
    }

    // You probably don't want to call this over multiple TX, since each call
    // recreates a tensor complex
    fn pre_rot_wect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        let device = tch::Device::cuda_if_available();
        let tensor_complex = WeightedTensorComplex::from(self, device);
        tensor_complex.pre_rot_wect(params, array2_to_tensor(&tx, device))
    }
}

impl<P, W> TensorEct for WeightedOptComplex<P, W>
where
    P: Element,
    W: Element,
{
    type RotMat = Array2<f64>;
    fn ect(&self, params: &ECTParams) -> Tensor {
        if self.has_missing_dims() {
            panic!("Cannot compute ECT with missing dimensions");
        }
        let device = tch::Device::cuda_if_available();
        let tensor_complex = TensorComplex::from_weighted(self, device);
        tensor_complex.ect(params)
    }

    // You probably don't want to call this over multiple TX, since each call
    // recreates a tensor complex
    fn pre_rot_ect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        let device = tch::Device::cuda_if_available();
        let tensor_complex = TensorComplex::from_weighted(self, device);
        tensor_complex.pre_rot_ect(params, array2_to_tensor(&tx, device))
    }
}

impl<P> TensorEct for OptComplex<P>
where
    P: Element,
{
    type RotMat = Array2<f64>;
    fn ect(&self, params: &ECTParams) -> Tensor {
        if self.has_missing_dims() {
            panic!("Cannot compute ECT with missing dimensions");
        }
        let device = tch::Device::cuda_if_available();
        let tensor_complex = TensorComplex::from(self, device);
        tensor_complex.ect(params)
    }

    // You probably don't want to call this over multiple TX, since each call
    // recreates a tensor complex
    fn pre_rot_ect(&self, params: &ECTParams, tx: Self::RotMat) -> Tensor {
        let device = tch::Device::cuda_if_available();
        let tensor_complex = TensorComplex::from(self, device);
        tensor_complex.pre_rot_ect(params, array2_to_tensor(&tx, device))
    }
}

fn vertex_indices(params: &ECTParams, vertex_coords: &Tensor) -> Tensor {
    let v_norms = vertex_coords.norm_scalaropt_dim(2, [1], false); // vertex l2 norms
    let max_height = v_norms.max();

    let v_heights: tch::Tensor = vertex_coords.matmul(&params.dirs.transpose(0, 1));
    let v_indices: tch::Tensor = ((&params.num_heights - 1 as i64) * (&max_height + v_heights)
        / (2.0 as f64 * &max_height))
        .ceil()
        .to_kind(Kind::Int64);
    v_indices
}

fn sparsify_index_tensor(params: &ECTParams, index_tensor: &Tensor, weight_dtype: Kind) -> Tensor {
    let n = index_tensor.size()[0];
    let device = index_tensor.device();
    let grid = tch::Tensor::meshgrid(&[
        Tensor::arange(n, (Kind::Int64, device)),
        Tensor::arange(params.dirs.size()[0], (Kind::Int64, device)),
    ]);
    let i = grid[0].flatten(0, -1);
    let j = grid[1].flatten(0, -1);
    let k = index_tensor.flatten(0, -1);
    let indices = Tensor::stack(&[&i, &j, &k], 1).transpose(0, 1);

    let values = Tensor::ones(i.size(), (weight_dtype, device));

    let shape = vec![
        n,
        params.dirs.size()[0],
        params.num_heights, //.int64_value(&[0]),
    ];

    Tensor::sparse_coo_tensor_indices_size(&indices, &values, &shape, (Kind::Float, device), false)
}

fn ect(complex: &TensorComplex, tx: Option<Tensor>, params: &ECTParams) -> Tensor {
    let vertex_coords = match tx {
        Some(tx) => complex.get_vertices().matmul(&tx.transpose(0, 1)),
        None => complex.get_vertices().shallow_clone(),
    };
    let v_indices = vertex_indices(&params, &vertex_coords);
    let v_graphs = sparsify_index_tensor(&params, &v_indices, Kind::Int64);
    let mut contributions = v_graphs.internal_sparse_sum_dim(vec![0]);

    for dim in 1..=complex.size() {
        let simplex_tensor = &complex.get_simplices_dim(dim);

        let v_pair_indices = Tensor::empty(
            // HACK: loop through and assign instead of advanced
            // indexing
            vec![
                simplex_tensor.size()[0],
                simplex_tensor.size()[1],
                v_indices.size()[1],
            ],
            (Kind::Int64, simplex_tensor.device()),
        );

        for i in 0..simplex_tensor.size()[0] {
            let indices = simplex_tensor.get(i);
            let indexed_values = v_indices.index_select(0, &indices);
            v_pair_indices.get(i).copy_(&indexed_values);
        }

        let simplex_indices = v_pair_indices.amax(&[1], false);

        let simplex_graphs = sparsify_index_tensor(&params, &simplex_indices, Kind::Int64);
        let simplex_contributions = simplex_graphs.internal_sparse_sum_dim(vec![0]);
        contributions += simplex_contributions * (-1i64).pow(dim as u32);
    }

    let ect = contributions.to_dense(None, false).cumsum(1, Kind::Float);
    ect
}
fn wect(complex: &WeightedTensorComplex, tx: Option<Tensor>, params: &ECTParams) -> Tensor {
    let vertex_coords = match tx {
        Some(tx) => complex.get_vertices().matmul(&tx.transpose(0, 1)),
        None => complex.get_vertices().shallow_clone(),
    };
    let vertex_weights = complex.get_weights_dim(0);
    let v_indices = vertex_indices(&params, &vertex_coords);
    let v_graphs = sparsify_index_tensor(&params, &v_indices, Kind::Float);
    let vertex_weights = vertex_weights.view([-1, 1, 1]);
    let weighted_v_graphs = vertex_weights * v_graphs;
    let mut contributions = weighted_v_graphs.internal_sparse_sum_dim(vec![0]);

    for dim in 1..=complex.size() {
        let simplex_tensor = &complex.get_simplices_dim(dim);

        let v_pair_indices = Tensor::empty(
            // HACK: loop through and assign instead of advanced
            // indexing
            vec![
                simplex_tensor.size()[0],
                simplex_tensor.size()[1],
                v_indices.size()[1],
            ],
            (Kind::Int64, simplex_tensor.device()),
        );

        for i in 0..simplex_tensor.size()[0] {
            let indices = simplex_tensor.get(i);
            let indexed_values = v_indices.index_select(0, &indices);
            v_pair_indices.get(i).copy_(&indexed_values);
        }

        let simplex_indices = v_pair_indices.amax(&[1], false);

        let simplex_graphs = sparsify_index_tensor(&params, &simplex_indices, Kind::Float);
        let simplex_weights = complex.get_weights_dim(dim).view([-1, 1, 1]);
        let weighted_simplex_graphs = simplex_weights * simplex_graphs;
        let simplex_contributions = weighted_simplex_graphs.internal_sparse_sum_dim(vec![0]);
        contributions += simplex_contributions * (-1.0f64).powi(dim as i32);
    }

    let wect = contributions.to_dense(None, false).cumsum(1, Kind::Float);
    wect
}
fn sample2d(num_dirs: i64, device: Device) -> Tensor {
    let t = Tensor::linspace(0.0, 6.283185, num_dirs, (Kind::Float, device));
    Tensor::stack(&[t.cos(), t.sin()], 1)
}

fn sample3d(num_dirs: i64, device: Device) -> Tensor {
    let _phi = (1.0 + 5.0f64.sqrt()) / 2.0;
    let z = Tensor::linspace(
        1.0 - 1.0 / num_dirs as f64,
        -1.0 + 1.0 / num_dirs as f64,
        num_dirs,
        (Kind::Float, device),
    );
    let theta = Tensor::linspace(0.0, 2.0 * 3.14159265359, num_dirs, (Kind::Float, device));
    let exp = Tensor::scalar_tensor(2.0, (Kind::Int64, device));
    let r: Tensor = (1.0 as f64 - z.pow(&exp)).sqrt();
    let x = &r * theta.cos();
    let y = &r * theta.sin();
    Tensor::stack(&[x, y, z], 1)
}

fn sample_dirs(num_dirs: i64, dim: i64, device: Device) -> Tensor {
    match dim {
        2 => sample2d(num_dirs, device),
        3 => sample3d(num_dirs, device),
        _ => panic!("Invalid dimension, no implementation for >3"),
    }
}
