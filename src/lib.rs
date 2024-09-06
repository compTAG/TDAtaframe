mod complex;
mod complex_interpolation;
pub mod complex_mapping;
mod io;
mod tensorwect;
mod utils;
// mod wect_interface;

pub mod expressions;
use pyo3::types::{PyModule, PyModuleMethods};
use pyo3::{pymodule, Bound, PyResult};
use pyo3_polars::PolarsAllocator;

#[pymodule]
fn _internal(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}

#[global_allocator]
static ALLOC: PolarsAllocator = PolarsAllocator::new();

#[cfg(test)]
mod tests {
    use crate::complex_mapping;
    use faer::{row, Mat, Row};
    use faer_ext::{IntoFaer, IntoNdarray};
    use ndarray::{array, Array2, ArrayView2, Axis};

    struct Octahedron {
        vertices: Array2<f32>,
        triangles: Array2<usize>,
        weights: Vec<f32>,
    }

    impl Octahedron {
        fn new(weighted: bool, x: f32, y: f32, z: f32) -> Octahedron {
            let vertices = array![
                [x, 0., 0.],
                [-x, 0., 0.],
                [0., y, 0.],
                [0., -y, 0.],
                [0., 0., z],
                [0., 0., -z]
            ];

            let triangles = array![
                [0, 2, 4],
                [2, 1, 4],
                [1, 3, 4],
                [3, 0, 4],
                [0, 2, 5],
                [2, 1, 5],
                [1, 3, 5],
                [3, 0, 5]
            ];
            let weights = if weighted {
                vec![-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0]
            } else {
                vec![1.0; 8]
            };

            Octahedron {
                vertices,
                triangles,
                weights,
            }
        }
    }

    #[test]
    fn test_compute_barycenters() {
        let vertices = array![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
        let simplices = array![[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]];

        let barycenters = complex_mapping::compute_barycenters(&vertices.view(), &simplices.view());
        let expected = array![
            [1. / 3., 1. / 3., 0.],
            [1. / 3., 0., 1. / 3.],
            [0., 1. / 3., 1. / 3.],
            [1. / 3., 1. / 3., 1. / 3.]
        ];
        dbg!(&barycenters);

        assert_eq!(barycenters, expected);

        let vertices2 = array![
            [0., 0., 0., 3.0],
            [1., 0., 0., 2.0],
            [0., 1., 0., 1.0],
            [0., 0., 1., 0.0]
        ];

        let simplices2 = array![[0, 1], [1, 3], [0, 2], [2, 3]];

        let barycenters2 =
            complex_mapping::compute_barycenters(&vertices2.view(), &simplices2.view());

        let expected2 = array![
            [0.5, 0.0, 0.0, 2.5],
            [0.5, 0.0, 0.5, 1.0],
            [0.0, 0.5, 0.0, 2.0],
            [0.0, 0.5, 0.5, 0.5]
        ];

        assert_eq!(barycenters2, expected2);
    }

    #[test]
    fn test_bubble_vt() {
        let oct = Octahedron::new(true, 2.0, 1.0, 3.0);
        let svd = complex_mapping::compute_vt(IntoFaer::into_faer(oct.vertices.view()));
        let svd = IntoNdarray::into_ndarray(svd.as_ref());
        let target = array![[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [-0.0, -1.0, 0.0]];

        assert_eq!(svd, target);
    }

    #[test]
    fn test_wo() {
        let oct = Octahedron::new(false, 2.0, 1.0, 3.0);
        let barycenters =
            complex_mapping::compute_barycenters(&oct.vertices.view(), &oct.triangles.view());

        let barymat = IntoFaer::into_faer(barycenters.view());
        let wo = complex_mapping::weighted_centroid_offset(barymat, &oct.weights);
        let target: Row<f32> = row![0.0, 0.0, 0.0];
        assert_eq!(wo, target);

        // TODO: test with weighted octahedron
        //let oct = Octahedron::new(true, 2.0, 1.0, 1.0);
        //let barycenters = complex_mapping::compute_barycenters(&oct.vertices.view(), &oct.triangles.view());
        //let wo = complex_mapping::weighted_centroid_offset(&barycenters, &oct.weights, &Array2::eye(3));
        //let target = array![0.0, 0.0, 0.0];
        //assert_eq!(wo, target);
    }
}
