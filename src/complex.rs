use ndarray::Array2;
use num_traits::float::Float;
use std::collections::HashSet;

use crate::complex_interpolation::interpolate_simplices_down;
use crate::complex_mapping::compute_maps_svd;
use crate::utils::array2_to_tensor;
use std::boxed::Box;
use std::rc::Rc;
use tch::{Device, IValue, Tensor};

pub struct WeightedComplex<V, S, W> {
    vertices: V,
    simplices: Vec<Option<S>>,
    weights: Vec<Option<W>>,
}
impl<V, S, W> WeightedComplex<V, S, W> {
    pub fn new(vertices: V) -> Self {
        Self {
            vertices,
            simplices: Vec::new(),
            weights: Vec::new(),
        }
    }

    // Create a new complex from a vertex set and a number of dimensions (dim = k for a k-dimensional complex).
    pub fn with_dims(vertices: V, dim: usize) -> Self {
        Self {
            vertices,
            simplices: Vec::with_capacity(dim),
            weights: Vec::with_capacity(dim + 1),
        }
    }

    // Create a new complex from a vertex set, a list of simplices, and a list of weights.
    // Panics if simplices.len() + 1 != weights.len(), or if the length of weights is 0.
    pub fn from_simplices(vertices: V, simplices: Vec<Option<S>>, weights: Vec<Option<W>>) -> Self {
        // loop over simplices and weights and push them into the complex
        // panic if lengths of simplices and weights do not match
        if simplices.len() + 1 != weights.len() {
            panic!("Lengths of simplices and weights do not match. Weights must be one dimension higher than simplices (to account for vertex weights).");
        }

        if weights.len() == 0 {
            panic!("Weights must be given for every simplex, got no weights for vertices (length of weights is 0");
        }

        // TODO: check that the number of simplices matches the number of weights in every dimension
        //
        Self {
            vertices,
            simplices,
            weights,
        }
    }

    // Return the top dimension of this simplicial complex.
    pub fn get_dim(&self) -> usize {
        self.simplices.len()
    }

    // Get the simplices of a given dimension greater than 0.
    pub fn get_simplices_dim(&self, dim: usize) -> Option<&Option<S>> {
        self.simplices.get(dim - 1)
    }

    pub fn get_weights_dim(&self, dim: usize) -> Option<&Option<W>> {
        self.weights.get(dim)
    }

    // Get the simplices and weights of a given dimension greater than 0.
    // Panics if the dimension is not found.
    pub fn get_pair_dim(&self, dim: usize) -> Option<(&Option<S>, &Option<W>)> {
        match (self.get_simplices_dim(dim), self.get_weights_dim(dim)) {
            (Some(simplices), Some(weights)) => Some((simplices, weights)),
            _ => None,
        }
    }

    pub fn get_vertices(&self) -> (&V, &Option<W>) {
        (&self.vertices, self.weights.get(0).unwrap())
    }

    pub fn set_simplices_dim(&mut self, simplices: Option<S>, dim: usize) {
        self.simplices[dim - 1] = simplices;
    }

    pub fn set_weights_dim(&mut self, weights: Option<W>, dim: usize) {
        self.weights[dim] = weights;
    }

    pub fn set_dim(&mut self, simplices: Option<S>, weights: Option<W>, dim: usize) {
        self.set_simplices_dim(simplices, dim);
        self.set_weights_dim(weights, dim);
    }

    pub fn set_vertex_weights(&mut self, weights: Option<W>) {
        self.weights[0] = weights;
    }

    // Push a pair of simplices and weights into the complex.
    // Panics if weights are given but not simplices (every weight must have a corresponding simplex).
    // Also panics if simplices and weights are pushed at the same time to different dimensions.
    pub fn push(&mut self, simplices: Option<S>, weights: Option<W>) {
        // check length of simplex and weight vectors before pushing
        if self.simplices.len() + 1 != self.weights.len() {
            panic!("Lengths of simplices and weights do not match. Currently, this complex has simplices up to dimension {} and weights up to dimension {}.", self.simplices.len(), self.weights.len() - 1);
        }

        // TODO: check that the dimension of the simplices matches the placement in the simplex
        // vector (i.e. if the simplices are being pushed into the 2nd dimension (simplices[1]), they should be 2-dimensional)

        // error if weights given but not simplices
        match (&simplices, &weights) {
            (None, Some(_)) => panic!("Weights given but not simplices"),
            _ => {
                // panic if lengths of simplices and weights do not match
                if self.simplices.len() != self.weights.len() {
                    panic!("Lengths of simplices and weights do not match");
                }
                self.simplices.push(simplices);
                self.weights.push(weights);
            }
        }
    }
}

pub type WeightedArrayComplex = WeightedComplex<Array2<f32>, Array2<usize>, Vec<f32>>;
pub type WeightedTensorComplex = WeightedComplex<Rc<Box<Tensor>>, Rc<Box<Tensor>>, Rc<Box<Tensor>>>;

impl<P: Float, F: Float> WeightedComplex<Array2<P>, Array2<usize>, Vec<F>> {
    // Interpolate the k-dimensional faces and weights of a complex from
    // a set of k+1 dimensional simplices and weights
    fn interpolate_down(&mut self, k: usize) {
        let simplices = self.get_simplices_dim(k + 1).unwrap().as_ref().unwrap();
        let weights = self.get_weights_dim(k + 1).unwrap().as_ref().unwrap();
        let (interp_simplices, interp_weights) = interpolate_simplices_down(&simplices, &weights);
        if k > 0 {
            self.set_dim(Some(interp_simplices), Some(interp_weights), k);
        } else if k == 0 {
            // any degenerate vertices will have 0 weight
            let mut vertex_weights = vec![F::zero(); self.vertices.shape()[0]];
            for i in 0..interp_simplices.shape()[0] {
                let vertex_as_index = interp_simplices[(i, 0)];
                vertex_weights[vertex_as_index] = interp_weights[i];
            }

            self.set_vertex_weights(Some(vertex_weights));

            if interp_simplices.shape()[0] != self.vertices.shape()[0] {
                println!("Warning: Extraneous vertices detected in complex");
            }
        } else {
            panic!("Can't interpolate simplices for dimension k={}", k);
        }
    }

    // Interpolate missing values in the complex.
    // Missing simplices are interpolated from the simplices of the next higher dimension.
    // Missing weights are interpolated from the weights of the next higher dimension.
    pub fn interpolate_missing_down(&mut self) {
        // gather missing simplex dimensions into a hash map of index -> hashmap<simplex_vector, (float, float)>
        let mut missing_simplices: HashSet<usize> = HashSet::new();

        // for each dimension, check if the simplices are missing
        (0..=self.get_dim()).for_each(|dim| {
            if self.get_simplices_dim(dim).is_none() {
                missing_simplices.insert(dim);
            }

            if self.get_weights_dim(dim).is_none() && !missing_simplices.contains(&dim) {
                missing_simplices.insert(dim);
            }
        });

        // iterate backwards thorough the missing simplices and weights and
        // interpolate the missing simplices and weights from the next higher dimension
        let mut missing_dims: Vec<usize> = missing_simplices.into_iter().map(|x| x).collect();
        missing_dims.sort();
        missing_dims.into_iter().rev().for_each(|k| {
            if k == self.get_dim() {
                panic!("Can't interpolate missing values for dimension k={}: the k+1 dimension simplices and weights are missing", k);
            }
            self.interpolate_down(k);
        });
    }

    // TODO: implement this
    // pub fn map_complex(&mut self) {
    //     let top_dim = self.get_dim();
    //     compute_maps_svd(
    //         self.get_vertices(),
    //         self.get_simplices_dim(top_dim).unwrap().as_ref().unwrap(),
    //         self.get_weights_dim(top_dim).unwrap().as_ref().unwrap(),
    //     );
    // }
    //
    pub fn get_embedded_dim(&self) -> usize {
        self.vertices.shape()[1]
    }

    // Get the number of simplices in a given dimension
    pub fn len(&self, dim: usize) -> Option<usize> {
        if dim == 0 {
            Some(self.vertices.len())
        } else {
            match self.get_simplices_dim(dim) {
                Some(result) => match result {
                    Some(simplices) => Some(simplices.len()),
                    None => None,
                },
                None => None,
            }
        }
    }
}
impl WeightedTensorComplex {
    pub fn zip_into_ival(&self) -> Vec<IValue> {
        let mut out: Vec<IValue> = Vec::with_capacity(self.weights.len());
        out.push(IValue::Tuple(vec![
            IValue::Tensor(self.vertices.shallow_clone()), // TODO: Instead, just allow it to use box
            IValue::Tensor(self.weights[0].as_ref().unwrap().shallow_clone()),
        ]));
        for (i, weight) in self.weights.iter().enumerate() {
            if i == 0 {
                continue;
            }
            out.push(IValue::Tuple(vec![
                IValue::Tensor(self.simplices[i - 1].as_ref().unwrap().shallow_clone()),
                IValue::Tensor(weight.as_ref().unwrap().shallow_clone()),
            ]));
        }
        out
    }

    pub fn from(complex: WeightedArrayComplex, device: Device) -> Self {
        let dim = complex.get_dim();

        let (vertices, vertex_weights) = complex.get_vertices();
        let vertices_t = Rc::new(Box::new(array2_to_tensor(vertices, device)));
        let mut weights = vec![Some(Rc::new(Box::new(Tensor::from_slice(
            &vertex_weights.as_ref().unwrap(),
        ))))];
        let mut simplices: Vec<Option<Rc<Box<Tensor>>>> = Vec::with_capacity(dim);

        (1..=dim).for_each(|k| {
            let (s, w) = complex.get_pair_dim(k).unwrap();
            let casted_simplices = s.as_ref().unwrap().mapv(|x| x as i64);

            let simplices_t = array2_to_tensor(&casted_simplices, device); // TODO: do we
                                                                           // manually cast usize to u32 or u64?
            simplices.push(Some(Rc::new(Box::new(simplices_t))));
            weights.push(Some(Rc::new(Box::new(Tensor::from_slice(
                &w.as_ref().unwrap(),
            )))));
        });

        Self::from_simplices(vertices_t, simplices, weights)
    }
}
