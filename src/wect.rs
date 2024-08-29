use tch::{kind, Device, Kind, Tensor};

#[derive(Debug)]
pub struct WECT {
    dirs: Tensor,
    num_heights: Tensor,
}

impl WECT {
    pub fn from_dirs(dirs: Tensor, num_heights: i64) -> Self {
        let device = dirs.device();
        let height_tensor = Tensor::scalar_tensor(num_heights, (kind::Kind::Int64, device));
        WECT {
            dirs: dirs.set_requires_grad(false),
            num_heights: height_tensor.set_requires_grad(false),
        }
    }

    pub fn new(embedded_dimension: i64, num_dirs: i64, num_heghts: i64, device: Device) -> Self {
        let dirs = sample_dirs(num_dirs, embedded_dimension, device);
        Self::from_dirs(dirs, num_heghts)
    }

    fn vertex_indices(&self, vertex_coords: &Tensor) -> Tensor {
        let v_norms = vertex_coords.norm_scalaropt_dim(2, [1], false); // vertex l2 norms
        let max_height = v_norms.max();

        let v_heights: tch::Tensor = vertex_coords.matmul(&self.dirs.transpose(0, 1));
        let v_indices: tch::Tensor = ((&self.num_heights - 1 as i64) * (&max_height + v_heights)
            / (2.0 as f64 * &max_height))
            .ceil()
            .to_kind(Kind::Int64);
        v_indices
    }

    fn sparsify_index_tensor(&self, index_tensor: &Tensor, weight_dtype: Kind) -> Tensor {
        let n = index_tensor.size()[0];
        let device = index_tensor.device();
        let grid = tch::Tensor::meshgrid(&[
            Tensor::arange(n, (Kind::Int64, device)),
            Tensor::arange(self.dirs.size()[0], (Kind::Int64, device)),
        ]);
        let i = grid[0].flatten(0, -1);
        let j = grid[1].flatten(0, -1);
        let k = index_tensor.flatten(0, -1);
        let indices = Tensor::stack(&[&i, &j, &k], 1);
        let values = Tensor::ones(&[i.size()[0]], (weight_dtype, device));
        Tensor::sparse_coo_tensor_indices_size(
            &indices.transpose(0, 1), // TODO: check if this is correct (i think so, indices is 2D)
            &values,
            &[n, self.dirs.size()[0], self.num_heights.int64_value(&[0])], // TODO: check if this is correct
            (Kind::Float, device),
            false,
        )
    }

    pub fn forward(&self, complex: Vec<(Tensor, Tensor)>) -> Tensor {
        let (vertex_coords, vertex_weights) = &complex[0];
        let v_indices = self.vertex_indices(&vertex_coords);
        let v_graphs = self.sparsify_index_tensor(&v_indices, Kind::Float);
        let vertex_weights = vertex_weights.view([-1, 1, 1]);
        let weighted_v_graphs = vertex_weights * v_graphs;
        let mut contributions = weighted_v_graphs.internal_sparse_sum_dim(vec![0]);

        for dim in 1..complex.len() {
            let simplex_tensor = &complex[dim].0;
            let v_pair_indices = v_indices.index_select(0, &simplex_tensor);
            let simplex_indices = v_pair_indices.amax(&[1], false);
            let simplex_graphs = self.sparsify_index_tensor(&simplex_indices, Kind::Float);
            let simplex_weights = complex[dim].1.view([-1, 1, 1]);
            let weighted_simplex_graphs = simplex_weights * simplex_graphs;
            let simplex_contributions = weighted_simplex_graphs.internal_sparse_sum_dim(vec![0]);
            contributions += simplex_contributions * (-1.0f64).powi(dim as i32);
        }

        let wect = contributions
            .to_dense(Kind::Float, false)
            .cumsum(1, Kind::Float);
        wect
    }
}

fn sample2D(num_dirs: i64, device: Device) -> Tensor {
    let t = Tensor::linspace(0.0, 6.283185, num_dirs, (Kind::Float, device));
    Tensor::stack(&[t.cos(), t.sin()], 1)
}

fn sample3D(num_dirs: i64, device: Device) -> Tensor {
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
        2 => sample2D(num_dirs, device),
        3 => sample3D(num_dirs, device),
        _ => panic!("Invalid dimension, no implementation for >3"),
    }
}
