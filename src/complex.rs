use ndarray::{Array2, ArrayView2};

pub trait SimplexList {
    fn shape(&self) -> Vec<usize>;
    fn len(&self) -> usize;
    fn dim(&self) -> usize;
}

impl<T> SimplexList for Array2<T> {
    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn dim(&self) -> usize {
        self.shape()[1] + 1
    }
}

impl SimplexList for Option<Array2<usize>> {
    fn shape(&self) -> Vec<usize> {
        self.as_ref().unwrap().shape().to_vec()
    }
    fn len(&self) -> usize {
        self.as_ref().unwrap().shape()[0]
    }

    fn dim(&self) -> usize {
        self.as_ref().unwrap().shape()[1] + 1
    }
}

impl<T> SimplexList for ArrayView2<'_, T> {
    fn shape(&self) -> Vec<usize> {
        self.shape().to_vec()
    }
    fn len(&self) -> usize {
        self.shape()[0]
    }

    fn dim(&self) -> usize {
        self.shape()[1] + 1
    }
}

impl SimplexList for Option<ArrayView2<'_, usize>> {
    fn shape(&self) -> Vec<usize> {
        self.as_ref().unwrap().shape().to_vec()
    }
    fn len(&self) -> usize {
        self.as_ref().unwrap().shape()[0]
    }

    fn dim(&self) -> usize {
        self.as_ref().unwrap().shape()[1] + 1
    }
}

impl SimplexList for Vec<Vec<f32>> {
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self.dim()]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self[0].len() + 1
    }
}

impl SimplexList for Vec<Vec<usize>> {
    fn shape(&self) -> Vec<usize> {
        vec![self.len(), self.dim()]
    }

    fn len(&self) -> usize {
        self.len()
    }

    fn dim(&self) -> usize {
        self[0].len() + 1
    }
}

pub trait Complex {
    type VRep;
    type SRep;

    // Return the dimension (k) of this simplicial complex.
    fn size(&self) -> usize;

    // Get the size of the embedded dimension
    fn vdim(&self) -> usize;

    // Get the number of simplices in a given dimension.
    fn len(&self, dim: usize) -> usize;

    // Get the simplices of a given dimension greater than 0.
    fn get_simplices_dim(&self, dim: usize) -> &Self::SRep;
    fn set_simplices_dim(&mut self, simplices: Self::SRep, dim: usize);

    fn get_vertices(&self) -> &Self::VRep;

    fn get_simplices(&self) -> &Vec<Self::SRep>;
}

pub trait Weighted: Complex {
    type WRep;

    fn from_simplices(
        vertices: <Self as Complex>::VRep,
        simplices: Vec<<Self as Complex>::SRep>,
        weights: Vec<Self::WRep>,
    ) -> Self;
    fn get_weights(&self) -> &Vec<Self::WRep>;
    fn get_weights_dim(&self, dim: usize) -> &Self::WRep;
    fn set_weights_dim(&mut self, weights: Self::WRep, dim: usize);

    fn get_pair_dim(&self, dim: usize) -> (&<Self as Complex>::SRep, &Self::WRep);
    fn set_dim(&mut self, simplices: <Self as Complex>::SRep, weights: Self::WRep, dim: usize);

    fn get_vertices_weights(&self) -> (&<Self as Complex>::VRep, &Self::WRep);
    fn set_vertex_weights(&mut self, weights: Self::WRep);
}

#[derive(Debug)]
pub struct SimplicialComplex<V, S> {
    pub vertices: V,
    pub simplices: Vec<S>,
}

impl<V, S> SimplicialComplex<V, S> {
    pub fn new(vertices: V) -> Self {
        Self {
            vertices,
            simplices: Vec::new(),
        }
    }
    pub fn with_dims(vertices: V, dim: usize) -> Self {
        Self {
            vertices,
            simplices: Vec::with_capacity(dim),
        }
    }

    pub fn from_simplices(vertices: V, simplices: Vec<S>) -> Self {
        Self {
            vertices,
            simplices,
        }
    }
}

impl<V, S> Complex for SimplicialComplex<V, S>
where
    V: SimplexList,
    S: SimplexList,
{
    type VRep = V;
    type SRep = S;

    fn size(&self) -> usize {
        self.simplices.len()
    }

    fn len(&self, dim: usize) -> usize {
        match dim {
            0 => self.vertices.len(),
            _ => self.get_simplices_dim(dim).len(),
        }
    }

    fn vdim(&self) -> usize {
        self.get_vertices().dim()
    }

    fn get_simplices_dim(&self, dim: usize) -> &S {
        if dim == 0 {
            panic!("Can't use get_simplices_dim for dimension 0");
        }
        &self.simplices.get(dim - 1).unwrap()
    }

    fn set_simplices_dim(&mut self, simplices: S, dim: usize) {
        self.simplices[dim - 1] = simplices;
    }

    fn get_vertices(&self) -> &V {
        &self.vertices
    }

    fn get_simplices(&self) -> &Vec<S> {
        &self.simplices
    }
}

#[derive(Debug)]
pub struct WeightedSimplicialComplex<V, S, W> {
    pub structure: SimplicialComplex<V, S>,
    pub weights: Vec<W>,
}

impl<V, S, W> WeightedSimplicialComplex<V, S, W> {
    pub fn new(vertices: V, weights: W) -> Self {
        Self {
            structure: SimplicialComplex::new(vertices),
            weights: vec![weights],
        }
    }

    pub fn with_dims(vertices: V, dim: usize, weights: W) -> Self {
        let mut new = Self {
            structure: SimplicialComplex::with_dims(vertices, dim),
            weights: Vec::with_capacity(dim + 1),
        };
        new.weights.push(weights);
        new
    }

    pub fn from_simplices(vertices: V, simplices: Vec<S>, weights: Vec<W>) -> Self {
        Self {
            structure: SimplicialComplex::from_simplices(vertices, simplices),
            weights,
        }
    }
}

impl<V, S, W> Complex for WeightedSimplicialComplex<V, S, W>
where
    V: SimplexList,
    S: SimplexList,
{
    type VRep = V;
    type SRep = S;

    fn size(&self) -> usize {
        self.structure.size()
    }

    fn len(&self, dim: usize) -> usize {
        self.structure.len(dim)
    }

    fn get_simplices_dim(&self, dim: usize) -> &S {
        self.structure.get_simplices_dim(dim)
    }

    fn set_simplices_dim(&mut self, simplices: S, dim: usize) {
        self.structure.set_simplices_dim(simplices, dim);
    }

    fn get_vertices(&self) -> &V {
        self.structure.get_vertices()
    }

    fn get_simplices(&self) -> &Vec<S> {
        self.structure.get_simplices()
    }

    fn vdim(&self) -> usize {
        self.structure.vdim()
    }
}

impl<V, S, W> Weighted for WeightedSimplicialComplex<V, S, W>
where
    V: SimplexList,
    S: SimplexList,
{
    type WRep = W;
    fn from_simplices(vertices: V, simplices: Vec<S>, weights: Vec<W>) -> Self {
        Self {
            structure: SimplicialComplex::from_simplices(vertices, simplices),
            weights,
        }
    }
    fn get_weights(&self) -> &Vec<W> {
        &self.weights
    }

    fn get_weights_dim(&self, dim: usize) -> &W {
        self.weights.get(dim).unwrap()
    }

    fn set_weights_dim(&mut self, weights: W, dim: usize) {
        self.weights[dim] = weights;
    }

    fn get_pair_dim(&self, dim: usize) -> (&S, &W) {
        (
            self.structure.get_simplices_dim(dim),
            self.weights.get(dim).unwrap(),
        )
    }

    fn set_dim(&mut self, simplices: S, weights: W, dim: usize) {
        self.structure.set_simplices_dim(simplices, dim);
        self.weights[dim] = weights;
    }

    fn get_vertices_weights(&self) -> (&V, &W) {
        (self.structure.get_vertices(), self.weights.get(0).unwrap())
    }

    fn set_vertex_weights(&mut self, weights: W) {
        self.weights[0] = weights;
    }
}

pub type WeightedArrayComplex = WeightedSimplicialComplex<Array2<f32>, Array2<usize>, Vec<f32>>;

//TODO: testing
fn test_new_cplex() {
    let vertices: Vec<Vec<f32>> = vec![
        vec![0.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 1.0],
    ];
    let simplices: Vec<Vec<Vec<usize>>> =
        vec![vec![vec![0, 1], vec![1, 2], vec![2, 3], vec![3, 0]]];
    let weights: Vec<Vec<f32>> = vec![vec![0.0, 1.0, 2.0, 3.0]];
    let cpl: WeightedSimplicialComplex<Vec<Vec<f32>>, Vec<Vec<usize>>, Vec<f32>> =
        WeightedSimplicialComplex::from_simplices(vertices, simplices, weights);
}
