extern crate ndarray as nd;

use std::fmt::Debug;

use faer::linalg::matmul::matmul;
use faer::solvers::{Svd, ThinSvd};
use faer::{
    col, row, ComplexField, Conjugate, Entity, Mat, MatRef, Parallelism, RealField, SimpleEntity,
};
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{array, Array1, Array2, ArrayView2, ScalarOperand};
use num_traits::{AsPrimitive, Float, FromPrimitive, One, ToPrimitive, Zero};
use rand::seq::SliceRandom;

use rand::thread_rng;

pub fn compute_barycenters<F: Float + FromPrimitive + Debug>(
    vertices: &ArrayView2<F>,
    simplices: &ArrayView2<usize>,
) -> Array2<F> {
    let mut barycenters = nd::Array2::<F>::zeros((simplices.shape()[0], vertices.shape()[1]));
    for (i, simplex) in simplices.outer_iter().enumerate() {
        let indices: Vec<usize> = simplex.to_vec().into_iter().map(|i| i as usize).collect();
        let barycenter = vertices
            .select(nd::Axis(0), &indices)
            .mean_axis(nd::Axis(0))
            .unwrap();
        barycenters.row_mut(i).assign(&barycenter);
    }
    return barycenters;
}

fn subsample_points_to_mat<F>(
    points: &ArrayView2<F>,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
) -> Mat<F>
where
    F: SimpleEntity + ComplexField,
{
    let n_points = points.shape()[0];
    if n_points < points.shape()[1] {
        return Mat::<F>::identity(points.shape()[1], points.shape()[1]);
    }
    let mut n_subsample = (n_points as f32 * subsample_ratio) // HACK: possible precision loss converting usize to f32
        .max(subsample_min as f32)
        .min(subsample_max as f32) as usize;
    if n_subsample > n_points {
        n_subsample = n_points
    }

    let mut indices: Vec<usize> = (0..n_points).collect();
    indices.shuffle(&mut thread_rng());
    let subsample_indices = &indices[0..n_subsample];
    let subsample_points = points.select(nd::Axis(0), subsample_indices);
    let submat: MatRef<F> = IntoFaer::into_faer(subsample_points.view());
    submat.to_owned()
}

pub fn compute_vt<F>(points: MatRef<F>) -> Mat<F>
where
    F: Float + SimpleEntity + ComplexField,
{
    let svd: ThinSvd<F> = points.thin_svd();
    let v_t = svd.v().transpose();
    v_t.to_owned()
}

pub fn compute_point_cloud_norm_factor<F>(points: &MatRef<F>) -> F
where
    F: Float + SimpleEntity + RealField,
    usize: AsPrimitive<F>,
{
    points
        .row_iter()
        .fold(F::zero(), |acc, row| acc + row.norm_l2())
        / (points.shape().0.as_())
}

pub fn weighted_centroid_offset<F>(
    points: MatRef<F>,
    weights: &Vec<F>,
    tx: MatRef<F>,
) -> row::Row<F>
where
    F: Float + ScalarOperand + SimpleEntity + RealField + ToPrimitive,
    usize: AsPrimitive<F>,
    f64: From<F>,
{
    let weight_col = col::from_slice(weights);

    let rotated = points * tx.transpose();

    let d: f64 = (points.shape().0.as_() * compute_point_cloud_norm_factor(&points)).into();
    let wcenter = (weight_col.transpose() * rotated) / d;

    wcenter
}

//TODO: generalize to n-dimensions
pub fn compute_heur_fix<F: Float + 'static>(tx: MatRef<F>, wcentroid: row::Row<F>) -> Vec<Array2<F>>
where
    F: SimpleEntity,
{
    let tx = IntoNdarray::into_ndarray(tx);
    let mut fix = Array2::<F>::eye(3);
    let x_is_pos = wcentroid[0] > F::zero();
    let y_is_pos = wcentroid[1] > F::zero();

    if x_is_pos && !y_is_pos {
        // then rotate x
        fix = array![
            [F::one(), F::zero(), F::zero()],
            [F::zero(), -F::one(), F::zero()],
            [F::zero(), F::zero(), -F::one()]
        ];
    } else if !x_is_pos && y_is_pos {
        // then rotate y
        fix = array![
            [-F::one(), F::zero(), F::zero()],
            [F::zero(), F::one(), F::zero()],
            [F::zero(), F::zero(), -F::one()]
        ];
    } else if !x_is_pos && !y_is_pos {
        // then rotate z
        fix = array![
            [-F::one(), F::zero(), F::zero()],
            [F::zero(), -F::one(), F::zero()],
            [F::zero(), F::zero(), F::one()]
        ];
    }

    vec![fix.dot(&tx)]
}

//TODO: generalize to n-dimensions
pub fn compute_copies<F: Float + 'static>(tx: MatRef<F>) -> Vec<Array2<F>>
where
    F: SimpleEntity,
{
    let iden = Array2::<F>::eye(3);
    let tx = IntoNdarray::into_ndarray(tx);
    let xpi = array![
        [F::one(), F::zero(), F::zero()],
        [F::zero(), -F::one(), F::zero()],
        [F::zero(), F::zero(), -F::one()]
    ];
    let ypi = array![
        [-F::one(), F::zero(), F::zero()],
        [F::zero(), F::one(), F::zero()],
        [F::zero(), F::zero(), -F::one()]
    ];
    let zpi = array![
        [-F::one(), F::zero(), F::zero()],
        [F::zero(), -F::one(), F::zero()],
        [F::zero(), F::zero(), F::one()]
    ];
    [iden, xpi, ypi, zpi]
        .iter()
        .map(|fix| fix.dot(&tx))
        .collect()
}

pub fn compute_maps_svd<F>(
    vertices: &ArrayView2<F>,
    simplices: &ArrayView2<usize>,
    face_normals: &Vec<F>,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<F>,
    copies: bool,
) -> Vec<Array2<F>>
where
    F: Float + FromPrimitive + RealField + SimpleEntity + ScalarOperand,
    usize: AsPrimitive<F>,
    f64: From<F>,
{
    let barycenters = compute_barycenters(&vertices.view(), &simplices.view());
    let bary_mat = IntoFaer::into_faer(barycenters.view());
    let sub_bary_mat = subsample_points_to_mat(
        &barycenters.view(),
        subsample_ratio,
        subsample_min,
        subsample_max,
    );
    let vt = compute_vt(sub_bary_mat.as_ref());
    let vtref = vt.as_ref();
    if copies {
        compute_copies(vtref)
    } else {
        match eps {
            Some(eps) => {
                let w_offset = weighted_centroid_offset(bary_mat, face_normals, vtref);
                if w_offset.norm_l2() < eps {
                    compute_copies(vtref)
                } else {
                    compute_heur_fix(vtref, w_offset)
                }
            }
            None => {
                if copies {
                    compute_copies(vtref)
                } else {
                    let vt = IntoNdarray::into_ndarray(vtref).into_owned();
                    vec![vt]
                }
            }
        }
    }
}
