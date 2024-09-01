use crate::{
    complex::{Complex, Weighted, WeightedOptComplex, WeightedTensorComplex},
    utils::array2_to_tensor,
};
use ndarray::Array2;
use tch::{kind::Element, Device, Kind, Tensor};

#[derive(Debug)]
pub struct WECTParams {
    pub dirs: Tensor,
    pub num_heights: i64,
}

impl WECTParams {
    pub fn from_dirs(dirs: Tensor, num_heights: i64) -> Self {
        // let device = dirs.device();
        // // let height_tensor = Tensor::scalar_tensor(num_heights, (kind::Kind::Int64, device));
        WECTParams {
            dirs: dirs.set_requires_grad(false),
            num_heights, //height_tensor.set_requires_grad(false),
        }
    }

    pub fn new(embedded_dimension: i64, num_dirs: i64, num_heghts: i64, device: Device) -> Self {
        let dirs = sample_dirs(num_dirs, embedded_dimension, device);
        Self::from_dirs(dirs, num_heghts)
    }
}
pub trait TensorWect {
    type RotMat;
    fn pre_rot_wect(&self, params: &WECTParams, tx: Self::RotMat) -> Tensor;
    fn wect(&self, params: &WECTParams) -> Tensor;
}

impl TensorWect for WeightedTensorComplex {
    type RotMat = Tensor;
    fn wect(&self, params: &WECTParams) -> Tensor {
        wect(self, self.get_vertices().shallow_clone(), params)
    }

    fn pre_rot_wect(&self, params: &WECTParams, tx: Self::RotMat) -> Tensor {
        let vertex_coords = self.get_vertices();
        let rotated_vertex_coords = vertex_coords.matmul(&tx.transpose(0, 1));
        wect(self, rotated_vertex_coords, params)
    }
}

impl<P, W> TensorWect for WeightedOptComplex<P, W>
where
    P: Element,
    W: Element,
{
    type RotMat = Array2<f64>;
    fn wect(&self, params: &WECTParams) -> Tensor {
        if self.has_missing_dims() {
            panic!("Cannot compute WECT with missing dimensions");
        }
        let device = tch::Device::cuda_if_available();
        let tensor_complex = WeightedTensorComplex::from(self, device);
        tensor_complex.wect(params)
    }

    // You probably don't want to call this over multiple TX, since each call
    // recreates a tensor complex
    fn pre_rot_wect(&self, params: &WECTParams, tx: Self::RotMat) -> Tensor {
        let device = tch::Device::cuda_if_available();
        let tensor_complex = WeightedTensorComplex::from(self, device);
        tensor_complex.pre_rot_wect(params, array2_to_tensor(&tx, device))
    }
}

fn vertex_indices(params: &WECTParams, vertex_coords: &Tensor) -> Tensor {
    let v_norms = vertex_coords.norm_scalaropt_dim(2, [1], false); // vertex l2 norms
    let max_height = v_norms.max();

    let v_heights: tch::Tensor = vertex_coords.matmul(&params.dirs.transpose(0, 1));
    let v_indices: tch::Tensor = ((&params.num_heights - 1 as i64) * (&max_height + v_heights)
        / (2.0 as f64 * &max_height))
        .ceil()
        .to_kind(Kind::Int64);
    v_indices
}

fn sparsify_index_tensor(params: &WECTParams, index_tensor: &Tensor, weight_dtype: Kind) -> Tensor {
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

fn wect(complex: &WeightedTensorComplex, vertex_coords: Tensor, params: &WECTParams) -> Tensor {
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
        (Kind::Float, Device::Cpu),
    );
    let theta = Tensor::linspace(
        0.0,
        2.0 * 3.14159265359,
        num_dirs,
        (Kind::Float, Device::Cpu),
    );
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
