use crate::complex::Complex;
use crate::complex::Weighted;
use crate::complex::WeightedOptComplex;
use ndarray::{Array2, Axis};
use num_traits::float::Float;
use std::collections::HashMap;
use std::collections::HashSet;

pub trait Interpolate {
    fn interpolate_missing_down(&mut self);
}

impl<P: Float, W: Float> Interpolate for WeightedOptComplex<P, W> {
    // Interpolate missing values in the complex.
    // Missing simplices are interpolated from the simplices of the next higher dimension.
    // Missing weights are interpolated from the weights of the next higher dimension.
    fn interpolate_missing_down(&mut self) {
        // gather missing simplex dimensions into a hash map of index -> hashmap<simplex_vector, (float, float)>
        let mut missing_simplices: HashSet<usize> = HashSet::new();

        if self.missing_weight_dim(0) {
            missing_simplices.insert(0);
        }
        // for each dimension, check if the simplices are missing
        (1..=self.size()).for_each(|dim| {
            if self.missing_simplex_dim(dim) {
                missing_simplices.insert(dim);
            }

            if self.missing_weight_dim(dim) && !missing_simplices.contains(&dim) {
                missing_simplices.insert(dim);
            }
        });

        // iterate backwards thorough the missing simplices and weights and
        // interpolate the missing simplices and weights from the next higher dimension
        let mut missing_dims: Vec<usize> = missing_simplices.into_iter().map(|x| x).collect();
        missing_dims.sort();
        missing_dims.into_iter().rev().for_each(|k| {
            if k == self.size() {
                panic!("Can't interpolate missing values for dimension k={}: the k+1 dimension simplices and weights are missing", k);
            }
            interpolate_down(self, k);
        });
    }
}

// Interpolate the k-dimensional faces and weights of a complex from
// a set of k+1 dimensional simplices and weights
fn interpolate_down<F, W>(complex: &mut WeightedOptComplex<F, W>, k: usize)
where
    F: Float,
    W: Float,
{
    let simplices: &Array2<usize> = &complex.get_simplices_dim(k + 1).as_ref().unwrap();
    let weights: &Vec<W> = complex.get_weights_dim(k + 1).as_ref().unwrap();
    let (interp_simplices, interp_weights) = interpolate_simplices_down(&simplices, &weights);
    if k > 0 {
        complex.set_dim(Some(interp_simplices), Some(interp_weights), k);
    } else if k == 0 {
        // any degenerate vertices will have 0 weight
        let mut vertex_weights = vec![W::zero(); complex.len(0)];
        for i in 0..interp_simplices.shape()[0] {
            let vertex_as_index = interp_simplices[(i, 0)];
            vertex_weights[vertex_as_index] = interp_weights[i];
        }

        complex.set_vertex_weights(Some(vertex_weights));

        if interp_simplices.shape()[0] != complex.len(0) {
            println!("Warning: Extraneous vertices detected in complex");
        }
    } else {
        panic!("Can't interpolate simplices for dimension k={}", k);
    }
}

// Returns the flattened faces of a simplex.
// To ensure that the faces are ordered, the vertices of
// the simplex must be sorted before calling this function.
fn get_ordered_faces(simplex: &Vec<usize>) -> Vec<Vec<usize>> {
    let k = simplex.len() - 1;
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(k + 1); // flattened list of faces

    for i in 0..(k + 1) {
        let mut face = simplex.clone();
        face.remove(i);
        faces.push(face);
    }
    faces
}

// TODO: DOCSTRING + parallelize
// Interpolate the faces and weights of a complex from a set of cofaces and weights
// This method only works to generate simplices of dimension 1 or higher.
// To generate the vertices, use the `interpolate_edges_down` method.
fn interpolate_simplices_down<F: Float>(
    simplices: &Array2<usize>,
    weights: &Vec<F>,
) -> (Array2<usize>, Vec<F>) {
    // A map from simplex to (cum_weights, num_weights)
    let n_simplices = simplices.shape()[0]; // the number of simplices
    let coface_dim = simplices.shape()[1] - 1; // the dimension of the simplices
    let face_dim = coface_dim - 1; // the dimension of the faces

    // initialize a hashmap to store the interpolated faces
    // the key is the face, the value is the sum of weights and the number of weights
    let mut interp_map: HashMap<Vec<usize>, (F, F)> = HashMap::new();

    // loop through all simplices
    simplices.axis_iter(Axis(0)).enumerate().for_each(|(i, s)| {
        let mut simplex = s.to_vec();
        simplex.sort(); // ensure simplex indicies are sorted for hash
        let faces: Vec<Vec<usize>> = get_ordered_faces(&simplex);

        // for each face, increment its weight from the coface
        faces.into_iter().for_each(|face| {
            if interp_map.contains_key(&face) {
                let (cum_weights, num_weights) = interp_map.get(&face).unwrap();
                interp_map.insert(face, (*cum_weights + weights[i], *num_weights + F::one()));
            } else {
                interp_map.insert(face, (weights[i], F::one()));
            }
        });
    });
    let face_weights: Vec<F> = interp_map.values().map(|(sum, num)| *sum / *num).collect();

    let mut face_list_flattened: Vec<usize> = Vec::with_capacity(interp_map.len() * (face_dim + 1));
    interp_map.into_keys().for_each(|mut face| {
        face_list_flattened.append(&mut face);
    });

    let num_faces = face_list_flattened.len() / (face_dim + 1);

    let faces =
        Array2::<usize>::from_shape_vec((num_faces, face_dim + 1), face_list_flattened).unwrap();
    (faces, face_weights)
}
