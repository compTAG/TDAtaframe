use std::fmt::Debug;

use faer::mat::Mat;
use faer::mat::MatRef;
use faer::{col, row};
use faer_entity::SimpleEntity;
use faer_ext::{IntoFaer, IntoNdarray};
use ndarray::{array, Array2, ArrayView2, Axis, ScalarOperand};
use num_traits::{AsPrimitive, Float, FromPrimitive, ToPrimitive};
use rand::seq::SliceRandom;

use rand::thread_rng;

use crate::complex::{Complex, Weighted};
use crate::complex_opt::WeightedOptComplex;
pub trait PreMappable {
    type Output;
    type Point;
    fn premap_copy(
        &self,
        align_dim: usize,
        subsample_ratio: f32,
        subsample_min: usize,
        subsample_max: usize,
        eps: Option<f64>,
        copies: bool,
    ) -> Vec<Self::Output>;

    fn premap(
        &self,
        align_dim: usize,
        subsample_ratio: f32,
        subsample_min: usize,
        subsample_max: usize,
    ) -> Self::Output;
}

impl<P> PreMappable for WeightedOptComplex<P, P>
where
    P: Float + FromPrimitive + faer::RealField + SimpleEntity + ScalarOperand,
    f64: From<P>,
    usize: AsPrimitive<P>,
{
    type Output = Array2<P>;
    type Point = P;

    fn premap_copy(
        &self,
        align_dim: usize,
        subsample_ratio: f32,
        subsample_min: usize,
        subsample_max: usize,
        eps: Option<f64>,
        copies: bool,
    ) -> Vec<Self::Output> {
        let (simplices, weights) = self.get_pair_dim(align_dim);
        compute_maps_svd_copies(
            &self.get_vertices().view(),
            &simplices.as_ref().unwrap().view(),
            weights.as_ref().unwrap(),
            subsample_ratio,
            subsample_min,
            subsample_max,
            eps,
            copies,
        )
    }

    fn premap(
        &self,
        align_dim: usize,
        subsample_ratio: f32,
        subsample_min: usize,
        subsample_max: usize,
    ) -> Self::Output {
        let (simplices, weights) = self.get_pair_dim(align_dim);
        compute_map_svd(
            &self.get_vertices().view(),
            &simplices.as_ref().unwrap().view(),
            weights.as_ref().unwrap(),
            subsample_ratio,
            subsample_min,
            subsample_max,
        )
    }
}

pub fn compute_barycenters<F: Float + FromPrimitive + Debug>(
    vertices: &ArrayView2<F>,
    simplices: &ArrayView2<usize>,
) -> Array2<F> {
    let mut barycenters = Array2::<F>::zeros((simplices.shape()[0], vertices.shape()[1]));
    for (i, simplex) in simplices.outer_iter().enumerate() {
        let indices: Vec<usize> = simplex.to_vec().into_iter().map(|i| i as usize).collect();
        let barycenter = vertices
            .select(Axis(0), &indices)
            .mean_axis(Axis(0))
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
    F: SimpleEntity + faer::ComplexField,
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
    let subsample_points = points.select(Axis(0), subsample_indices);
    let submat: MatRef<F> = IntoFaer::into_faer(subsample_points.view());
    submat.to_owned()
}

pub fn compute_vt<F>(points: MatRef<F>) -> Mat<F>
where
    F: Float + SimpleEntity + faer::ComplexField,
{
    let svd = points.thin_svd();
    let v_t = svd.v().transpose();
    v_t.to_owned()
}

pub fn compute_point_cloud_norm_factor<F>(points: &MatRef<F>) -> F
where
    F: Float + SimpleEntity + faer::RealField,
    usize: AsPrimitive<F>,
{
    points
        .row_iter()
        .fold(F::zero(), |acc, row| acc + row.norm_l2())
        / (points.shape().0.as_())
}

pub fn weighted_centroid_offset<F>(points: MatRef<F>, weights: &Vec<F>) -> row::Row<F>
where
    F: Float + ScalarOperand + SimpleEntity + faer::RealField + ToPrimitive,
    usize: AsPrimitive<F>,
    f64: From<F>,
{
    let weight_col = col::from_slice(weights);

    let d: f64 = (points.shape().0.as_() * compute_point_cloud_norm_factor(&points)).into();
    let wcenter = (weight_col.transpose() * points) / d;

    wcenter
}

// place 1s or -1s on the diagonal of the matrix,
// depending on the sign of the weighted centroid offset
pub fn compute_heur_fix<F: Float + 'static>(wcentroid: col::Col<F>) -> Array2<F>
where
    F: SimpleEntity,
{
    let mut iden = Array2::<F>::eye(wcentroid.nrows());
    wcentroid.iter().enumerate().for_each(|(i, x)| {
        if x.is_sign_negative() {
            iden[[i, i]] = -F::one();
        }
    });
    iden
}

//TODO: generalize to n-dimensions
pub fn compute_copies<'a, F>(tx: MatRef<'a, F>) -> Vec<Array2<F>>
where
    F: 'a + Float + SimpleEntity,
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

pub fn compute_maps_svd_copies<F>(
    vertices: &ArrayView2<F>,
    simplices: &ArrayView2<usize>,
    simplex_weights: &Vec<F>,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
    eps: Option<f64>, // try heur fix if the weighted centroid offset is less than eps
    copies: bool,     // always produce copies
) -> Vec<Array2<F>>
where
    F: Float + FromPrimitive + faer::RealField + SimpleEntity + ScalarOperand,
    usize: AsPrimitive<F>,
    f64: From<F>,
    F: Into<f64>,
{
    // obtain subsampled barycenters
    let barycenters = compute_barycenters(&vertices.view(), &simplices.view());
    let bary_mat = IntoFaer::into_faer(barycenters.view());
    let sub_bary_mat = subsample_points_to_mat(
        &barycenters.view(),
        subsample_ratio,
        subsample_min,
        subsample_max,
    );

    // get the vt matrix from the svd of the subsampled barycenters
    let vt = compute_vt(sub_bary_mat.as_ref());
    let vtref = vt.as_ref();

    if copies {
        compute_copies(vtref)
    } else {
        match eps {
            Some(eps) => {
                let w_offset =
                    vtref * weighted_centroid_offset(bary_mat, simplex_weights).transpose();
                if w_offset.norm_l2().into() < eps {
                    compute_copies(vtref)
                } else {
                    let mut tx = IntoNdarray::into_ndarray(vtref).into_owned();
                    tx = compute_heur_fix(w_offset).dot(&tx);
                    vec![tx]
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

pub fn compute_map_svd<F>(
    vertices: &ArrayView2<F>,
    simplices: &ArrayView2<usize>,
    simplex_weights: &Vec<F>,
    subsample_ratio: f32,
    subsample_min: usize,
    subsample_max: usize,
) -> Array2<F>
where
    F: Float + FromPrimitive + faer::RealField + SimpleEntity + ScalarOperand,
    usize: AsPrimitive<F>,
    f64: From<F>,
{
    // obtain subsampled barycenters
    let barycenters = compute_barycenters(&vertices.view(), &simplices.view());
    let bary_mat = IntoFaer::into_faer(barycenters.view());
    let sub_bary_mat = subsample_points_to_mat(
        &barycenters.view(),
        subsample_ratio,
        subsample_min,
        subsample_max,
    );

    // get the vt matrix from the svd of the subsampled barycenters
    let vt = compute_vt(sub_bary_mat.as_ref());
    let vtref = vt.as_ref();

    let w_offset = vtref * weighted_centroid_offset(bary_mat, simplex_weights).transpose();
    let tx = IntoNdarray::into_ndarray(vtref).into_owned();
    compute_heur_fix(w_offset).dot(&tx)
}
