use crate::activations::{ActivationFunc, Sigmoid, RELU};
use crate::layers::node::Node;
use crate::math::{cut_val, dot};
use rand::Rng;
use std::rc::Rc;

pub trait Layer {
    fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>;
    fn backwards(&mut self, inputs: &Vec<f32>) -> Vec<f32>;
}

pub struct LayerDense {
    nodes: Vec<Node>,
    input_nodes_count: usize,
    nodes_count: usize,
    inputs: Vec<f32>,
}

impl LayerDense {
    pub fn new(
        input_nodes_count: usize,
        nodes_count: usize,
        activation: Rc<dyn ActivationFunc>,
    ) -> Self {
        Self {
            input_nodes_count,
            nodes_count,
            inputs: vec![],
            nodes: vec![0; nodes_count]
                .iter()
                .map(|_| Node::new(input_nodes_count, activation.clone(), None))
                .collect(),
        }
    }

    pub fn outputs(&self) -> Vec<f32> {
        self.nodes.iter().map(|node| node.output).collect()
    }
}

impl Layer for LayerDense {
    fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        self.inputs = inputs.clone();
        self.nodes
            .iter_mut()
            .map(|node| node.forward(inputs))
            .collect()
    }

    fn backwards(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
        let learning_rate = 0.1;
        let mut derivatives_for_input_nodes = vec![0.; self.input_nodes_count];
        for i in 0..self.input_nodes_count {
            derivatives_for_input_nodes[i] = self
                .nodes
                .iter()
                .enumerate()
                .map(|(j, node)| {
                    node.activation.compute_derivative(node.output)
                        * node.weights[i]
                        * derivatives[j]
                })
                .sum();
        }

        for i in 0..self.nodes_count {
            let node_ders = self.nodes[i]
                .compute_derivatives(&self.inputs, derivatives[i])
                .iter()
                .map(|val| val * learning_rate)
                .collect();
            self.nodes[i].update_weights(node_ders);
        }

        derivatives_for_input_nodes
    }
}

//pub struct LayerSoftmax {
//input_nodes_count: usize,
//inputs: Vec<f32>,
//pub outputs: Vec<f32>,
//}

//impl LayerSoftmax {
//pub fn new(input_nodes_count: usize) -> Self {
//Self {
//input_nodes_count,
//inputs: vec![],
//outputs: vec![],
//}
//}
//}

//impl Layer for LayerSoftmax {
//fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
//self.inputs = inputs.clone();
//self.outputs = softmax(inputs);
//self.outputs.clone()
//}

//fn backwards(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
//softmax_derivative(&self.inputs)
//.iter()
//.zip(derivatives)
//.map(|(sd, d)| {
//if sd.is_nan() {
//return 0.1 * d;
//}
//sd * d
//})
//.collect()
//}
//}
