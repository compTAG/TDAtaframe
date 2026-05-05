use crate::complex::{
    Complex, SimplexList, SimplicialComplex, Weighted, WeightedSimplicialComplex,
};
use crate::complex_opt::OptComplex;
use crate::complex_opt::WeightedOptComplex;
use crate::utils::array2_to_tensor;
use std::rc::Rc;
use tch::{Device, Tensor};

type BTensor = Rc<Box<Tensor>>;
impl SimplexList for BTensor {
    fn shape(&self) -> Vec<usize> {
        self.size().into_iter().map(|x| x as usize).collect()
    }

    fn len(&self) -> usize {
        self.size()[0] as usize
    }

    fn dim(&self) -> usize {
        self.size()[1] as usize + 1
    }
}
pub type WeightedTensorComplex = WeightedSimplicialComplex<BTensor, BTensor, BTensor>;
impl WeightedSimplicialComplex<BTensor, BTensor, BTensor> {
    // pub fn zip_into_ival(&self) -> Vec<IValue> {
    //     let mut out: Vec<IValue> = Vec::with_capacity(self.weights.len());
    //     out.push(IValue::Tuple(vec![
    //         IValue::Tensor(self.get_vertices().shallow_clone()), // TODO: Instead, just allow it to use box
    //         IValue::Tensor(self.weights[0].shallow_clone()),
    //     ]));
    //     for (i, weight) in self.weights.iter().enumerate() {
    //         if i == 0 {
    //             continue;
    //         }
    //         out.push(IValue::Tuple(vec![
    //             IValue::Tensor(self.get_simplices_dim(i).shallow_clone()),
    //             IValue::Tensor(weight.shallow_clone()),
    //         ]));
    //     }
    //     out
    // }

    pub fn from<V, W>(complex: &WeightedOptComplex<V, W>, device: Device) -> Self
    where
        V: tch::kind::Element,
        W: tch::kind::Element,
    {
        let dim = complex.size();

        let (vertices, vertex_weights) = complex.get_vertices_weights();
        let vertices_t = Rc::new(Box::new(array2_to_tensor(vertices, device)));

        let weights_silce: &[W] = &vertex_weights.as_ref().unwrap();

        let mut weights = vec![Rc::new(Box::new(
            Tensor::from_slice(weights_silce).to(device),
        ))];
        let mut simplices: Vec<Rc<Box<Tensor>>> = Vec::with_capacity(dim);

        (1..=dim).for_each(|k| {
            let (s, w) = complex.get_pair_dim(k);
            let casted_simplices = s.as_ref().unwrap().mapv(|x| x as i64);

            let simplices_t = array2_to_tensor(&casted_simplices, device); // TODO: do we
                                                                           // manually cast usize to u32 or u64?
            simplices.push(Rc::new(Box::new(simplices_t)));
            weights.push(Rc::new(Box::new(
                Tensor::from_slice(w.as_ref().unwrap()).to(device),
            )));
        });

        Self::from_simplices(vertices_t, simplices, weights)
    }
}

pub type TensorComplex = SimplicialComplex<BTensor, BTensor>;
impl SimplicialComplex<BTensor, BTensor> {
    pub fn from<V>(complex: &OptComplex<V>, device: Device) -> Self
    where
        V: tch::kind::Element,
    {
        let dim = complex.size();

        let vertices = complex.get_vertices();
        let vertices_t = Rc::new(Box::new(array2_to_tensor(vertices, device)));

        let mut simplices: Vec<Rc<Box<Tensor>>> = Vec::with_capacity(dim);

        (1..=dim).for_each(|k| {
            let s = complex.get_simplices_dim(k);
            let casted_simplices = s.as_ref().unwrap().mapv(|x| x as i64);

            let simplices_t = array2_to_tensor(&casted_simplices, device); // TODO: do we
                                                                           // manually cast usize to u32 or u64?
            simplices.push(Rc::new(Box::new(simplices_t)));
        });

        Self::from_simplices(vertices_t, simplices)
    }

    pub fn from_weighted<V, W>(complex: &WeightedOptComplex<V, W>, device: Device) -> Self
    where
        V: tch::kind::Element,
        W: tch::kind::Element,
    {
        Self::from::<V>(&complex.structure, device)
    }

    // // Add weights to an existing tensor complex
    // pub fn to_weighted(
    //     &self,
    //     weights: Vec<Rc<Box<Tensor>>>,
    //     device: Device,
    // ) -> WeightedTensorComplex {
    //     // Ensure each provided weight tensor is transferred to the given device.
    //     let weights_on_device: Vec<Rc<Box<Tensor>>> = weights
    //         .into_iter()
    //         .map(|w| Rc::new(Box::new((**w).to(device))))
    //         .collect();
    //         TODO : Broken, weights doesn't include the vertex weights
    //
    //     WeightedTensorComplex::from_simplices(
    //         self.get_vertices().clone(),
    //         self.get_simplices().clone(),
    //         weights_on_device,
    //     )
    // }

    // Add weights of value 1 to an existing tensor complex
    pub fn to_weighted_ones(&self, device: Device) -> WeightedTensorComplex {
        let kind = self.get_vertices().kind();
        let mut weights: Vec<Rc<Box<Tensor>>> = Vec::with_capacity(self.get_simplices().len() + 1);
        // Add the vertex weights
        weights.push(Rc::new(Box::new(Tensor::ones(
            self.get_vertices().size()[0],
            (kind, device),
        ))));
        self.get_simplices().iter().for_each(|s| {
            // Add the weights for each simplex
            weights.push(Rc::new(Box::new(Tensor::ones(s.size()[0], (kind, device)))));
        });

        let wtc = WeightedTensorComplex::from_simplices(
            self.get_vertices().clone(),
            self.get_simplices().clone(),
            weights,
        );
        println!("{}", wtc.get_vertices());
        println!("{}", wtc.get_weights()[0]);
        println!("{}", wtc.get_simplices_dim(1));
        println!("{}", wtc.get_weights()[1]);
        wtc
    }
}
