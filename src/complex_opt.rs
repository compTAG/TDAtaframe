use crate::complex::{Complex, SimplicialComplex, Weighted, WeightedSimplicialComplex};
use ndarray::Array2;
use ndarray::Axis;
use num_traits::float::Float;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

pub type WeightedOptComplex<P, W> =
    WeightedSimplicialComplex<Array2<P>, Option<Array2<usize>>, Option<Vec<W>>>;
pub type OptComplex<P> = SimplicialComplex<Array2<P>, Option<Array2<usize>>>;

impl<P> OptComplex<P> {
    pub fn missing_simplex_dim(&self, dim: usize) -> bool {
        self.simplices[dim - 1].is_none()
    }

    pub fn has_missing_dims(&self) -> bool {
        (1..=self.size()).any(|x| self.missing_simplex_dim(x))
    }

    pub fn from_provided(
        vertices: Array2<P>,
        mut simplices: VecDeque<Array2<usize>>,
        psimps: &Vec<usize>,
    ) -> Self {
        if psimps.len() == 0 && psimps[0] == 0 {
            panic!("Provided simplices must be greater than 0");
        }
        let k = psimps.get(psimps.len() - 1).unwrap(); // the top dimension

        let mut opt_simplices: Vec<Option<Array2<usize>>> = Vec::with_capacity(*k);

        let mut i = 1; // i tracks the current dimension
        psimps.iter().for_each(|&dim| {
            while i < dim {
                // fill missing dimensions
                opt_simplices.push(None);
                i += 1;
            }

            let simplex_array = simplices.pop_front().unwrap();
            opt_simplices.push(Some(simplex_array.into_owned()));
            i += 1;
        });

        Self::from_simplices(vertices, opt_simplices)
    }
}

impl<P, W> WeightedOptComplex<P, W> {
    pub fn missing_simplex_dim(&self, dim: usize) -> bool {
        self.structure.simplices[dim - 1].is_none()
    }

    pub fn missing_weight_dim(&self, dim: usize) -> bool {
        self.weights[dim].is_none()
    }

    pub fn has_missing_dims(&self) -> bool {
        (1..=self.size()).any(|x| self.missing_simplex_dim(x) || self.missing_weight_dim(x))
    }

    pub fn from_provided(
        vertices: Array2<P>,
        simplices: VecDeque<Array2<usize>>,
        mut weights: VecDeque<Vec<W>>,
        psimps: &Vec<usize>,
        pweights: &Vec<usize>,
    ) -> Self {
        let k = psimps.get(psimps.len() - 1).unwrap(); // the top dimension
        let structure = OptComplex::from_provided(vertices, simplices, psimps);

        let mut opt_weights: Vec<Option<Vec<W>>> = Vec::with_capacity(k + 1);

        let mut i = 0;
        pweights.iter().for_each(|&dim| {
            while i < dim {
                opt_weights.push(None);
                i += 1;
            }
            opt_weights.push(Some(weights.pop_front().unwrap()));
            i += 1;
        });
        Self::from_structure(structure, opt_weights)
    }
}

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
            interpolate_weighted_down(self, k);
        });
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

// Interpolate the k-dimensional faces and weights of a complex from
// a set of k+1 dimensional simplices and weights
fn interpolate_weighted_down<F, W>(complex: &mut WeightedOptComplex<F, W>, k: usize)
where
    F: Float,
    W: Float,
{
    let simplices: &Array2<usize> = &complex.get_simplices_dim(k + 1).as_ref().unwrap();
    let weights: &Vec<W> = complex.get_weights_dim(k + 1).as_ref().unwrap();
    let (interp_simplices, interp_weights) =
        interpolate_weighted_simplices_down(&simplices, &weights);
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

// TODO: DOCSTRING + parallelize
// Interpolate the faces and weights of a complex from a set of cofaces and weights
// This method only works to generate simplices of dimension 1 or higher.
// To generate the vertices, use the `interpolate_edges_down` method.
fn interpolate_weighted_simplices_down<F: Float>(
    simplices: &Array2<usize>,
    weights: &Vec<F>,
) -> (Array2<usize>, Vec<F>) {
    // A map from simplex to (cum_weights, num_weights)
    // let n_simplices = simplices.shape()[0]; // the number of simplices
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

impl<P: Float> Interpolate for OptComplex<P> {
    // Interpolate missing values in the complex.
    // Missing simplices are interpolated from the simplices of the next higher dimension.
    fn interpolate_missing_down(&mut self) {
        // gather the relevant dimensions
        let mut missing_simplices: HashSet<usize> = HashSet::new();

        // for each dimension, check if the simplices are missing
        (1..=self.size()).for_each(|dim| {
            if self.missing_simplex_dim(dim) {
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
fn interpolate_down<F>(complex: &mut OptComplex<F>, k: usize)
where
    F: Float,
{
    let simplices: &Array2<usize> = &complex.get_simplices_dim(k + 1).as_ref().unwrap();
    let interp_simplices = interpolate_simplices_down::<F>(&simplices);
    if k > 0 {
        complex.set_simplices_dim(Some(interp_simplices), k);
    } else if k == 0 {
        if interp_simplices.shape()[0] != complex.len(0) {
            println!("Warning: Extraneous vertices detected in complex");
        }
    } else {
        panic!("Can't interpolate simplices for dimension k={}", k);
    }
}

// TODO: DOCSTRING + parallelize
// Interpolate the faces and weights of a complex from a set of cofaces and weights
// This method only works to generate simplices of dimension 1 or higher.
// To generate the vertices, use the `interpolate_edges_down` method.
fn interpolate_simplices_down<F: Float>(simplices: &Array2<usize>) -> Array2<usize> {
    // A map from simplex to (cum_weights, num_weights)
    // let n_simplices = simplices.shape()[0]; // the number of simplices
    let coface_dim = simplices.shape()[1] - 1; // the dimension of the simplices
    let face_dim = coface_dim - 1; // the dimension of the faces

    // initialize a hashmap to store the interpolated faces
    // the key is the face, the value is the sum of weights and the number of weights
    let mut interp_map: HashSet<Vec<usize>> = HashSet::new();

    // loop through all simplices
    simplices.axis_iter(Axis(0)).for_each(|s| {
        let mut simplex = s.to_vec();
        simplex.sort(); // ensure simplex indicies are sorted for hash
        let faces: Vec<Vec<usize>> = get_ordered_faces(&simplex);

        // add each face
        faces.into_iter().for_each(|face| {
            if !interp_map.contains(&face) {
                interp_map.insert(face);
            }
        });
    });

    let mut face_list_flattened: Vec<usize> = Vec::with_capacity(interp_map.len() * (face_dim + 1));
    interp_map.into_iter().for_each(|mut face| {
        face_list_flattened.append(&mut face);
    });

    let num_faces = face_list_flattened.len() / (face_dim + 1);

    Array2::<usize>::from_shape_vec((num_faces, face_dim + 1), face_list_flattened).unwrap()
}
