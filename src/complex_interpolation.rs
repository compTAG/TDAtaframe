use ndarray::{Array2, Axis};
use num_traits::float::Float;
use std::collections::HashMap;

// Returns the flattened faces of a simplex.
// To ensure that the faces are ordered, the vertices of
// the simplex must be sorted before calling this function.
fn get_ordered_faces(simplex: &Vec<usize>) -> Vec<Vec<usize>> {
    let k = simplex.len() - 1;
    let mut faces: Vec<Vec<usize>> = Vec::with_capacity(k + 1); // flattened list of faces

    for i in 0..(k + 1) {
        let mut face = simplex.clone();
        face.remove(i);
        faces[i] = face;
    }
    faces
}

// TODO: DOCSTRING + parallelize
// Interpolate the faces and weights of a complex from a set of cofaces and weights
// This method only works to generate simplices of dimension 1 or higher.
// To generate the vertices, use the `interpolate_edges_down` method.
pub fn interpolate_simplices_down<F: Float>(
    simplices: &Array2<usize>,
    weights: &Vec<F>,
) -> (Array2<usize>, Vec<F>) {
    // A map from simplex to (cum_weights, num_weights)
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

    let faces =
        Array2::<usize>::from_shape_vec((coface_dim + 1, face_dim + 1), face_list_flattened)
            .unwrap();
    (faces, face_weights)
}
