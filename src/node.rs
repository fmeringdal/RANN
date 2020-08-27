use crate::activations::{ActivationFunc, Sigmoid, RELU};
use rand::Rng;
use std::rc::Rc;

pub struct Node {
    pub inputs: Vec<f32>,
    pub output: f32,
    pub weights: Vec<f32>,
    pub activation: Rc<dyn ActivationFunc>,
    pub number_of_inputs: usize,
}

impl Node {
    pub fn new(
        number_of_inputs: usize,
        activation: Rc<dyn ActivationFunc>,
        default_weights: Option<Vec<f32>>,
    ) -> Self {
        Self {
            weights: match default_weights {
                Some(ws) => ws,
                None => Self::gen_weights(number_of_inputs),
            },
            output: 0.,
            inputs: vec![],
            number_of_inputs,
            activation,
        }
    }

    fn gen_weights(count: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        vec![0; count].iter().map(|_| rng.gen::<f32>()).collect()
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> f32 {
        self.inputs = inputs.clone();
        let output = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        let output = self.activation.compute(output);
        self.output = output;
        output
    }

    pub fn compute_derivatives(&self, dc_da: f32) -> Vec<f32> {
        let part1_update = dc_da * self.activation.compute_derivative(self.output);
        let mut weight_derivatives = vec![0.; self.number_of_inputs];
        for i in 0..self.number_of_inputs {
            weight_derivatives[i] = part1_update * self.inputs[i];
        }
        weight_derivatives
    }

    pub fn update_weights(&mut self, delta_weights: Vec<f32>) {
        for i in 0..delta_weights.len() {
            self.weights[i] -= delta_weights[i];
        }
    }
}

pub struct LayerDense {
    nodes: Vec<Node>,
    input_nodes_count: usize,
    nodes_count: usize,
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
            nodes: vec![0; nodes_count]
                .iter()
                .map(|_| Node::new(input_nodes_count, activation.clone(), None))
                .collect(),
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        self.nodes
            .iter_mut()
            .map(|node| node.forward(inputs))
            .collect()
    }

    pub fn backwards(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
        // derivatives = [dc_da1, dc_da2]
        let learning_rate = 0.01;
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
                .compute_derivatives(derivatives[i])
                .iter()
                .map(|val| val * learning_rate)
                .collect();
            self.nodes[i].update_weights(node_ders);
        }

        derivatives_for_input_nodes
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn node_forward() {
        let mut node = Node::new(3, Rc::new(RELU::new()), Some(vec![1., 2., 3.]));
        let output = node.forward(&vec![1., 2., 3.]);
        assert_eq!(output, 14.);
    }

    #[test]
    fn node_derivatives_perfect_weights() {
        let mut node = Node::new(3, Rc::new(RELU::new()), Some(vec![1., 2., 3.]));
        let output = node.forward(&vec![1., 2., 3.]);
        let dervs = node.compute_derivatives(0.);
        assert_eq!(dervs, vec![0., 0., 0.]);
    }

    #[test]
    fn node_derivatives() {
        let mut node = Node::new(3, Rc::new(Sigmoid::new()), Some(vec![0.1, 0.3, 0.5]));
        let output = node.forward(&vec![1., 4., 5.]);
        let dervs = node.compute_derivatives(0.1502);
        assert_eq!(dervs, vec![0.002, 0.0079, 0.0099]);
    }

    #[test]
    fn node_compose() {
        let mut node1 = Node::new(3, Rc::new(Sigmoid::new()), None);
        let mut node2 = Node::new(3, Rc::new(Sigmoid::new()), None);
        let mut node3 = Node::new(2, Rc::new(Sigmoid::new()), None);
        let input = vec![1.1, 1.2, 1.3];
        let target = 0.25;
        for _ in 0..100000 {
            let output1 = node1.forward(&input);
            let output2 = node2.forward(&input);
            let output3 = node3.forward(&vec![output1, output2]);
            let error_delta = output3 - target;
            let node3de = node3.compute_derivatives(error_delta);
            let common = node3.activation.compute_derivative(node3.output) * error_delta;
            let dc_da11 = common * node3.weights[0];
            let dc_da12 = common * node3.weights[1];
            let node1de = node1.compute_derivatives(dc_da11);
            let node2de = node2.compute_derivatives(dc_da12);
            let learning_rate = 0.01;
            node1.update_weights(node1de.iter().map(|d| d * learning_rate).collect());
            node2.update_weights(node2de.iter().map(|d| d * learning_rate).collect());
            node3.update_weights(node3de.iter().map(|d| d * learning_rate).collect());
        }

        let output1 = node1.forward(&input);
        let output2 = node2.forward(&input);
        let output3 = node3.forward(&vec![output1, output2]);
        let error_delta = output3 - target;
        assert!(error_delta < 0.002);
    }

    #[test]
    fn node_layer() {
        let mut layer1 = LayerDense::new(3, 3, Rc::new(Sigmoid::new()));
        let mut layer2 = LayerDense::new(3, 1, Rc::new(Sigmoid::new()));
        let input = vec![1.1, 1.2, 1.3];
        let target = 0.25;
        for _ in 0..100000 {
            let outp1 = layer1.forward(&input);
            let outp2 = layer2.forward(&outp1);
            let error = outp2[0] - target;
            println!("Error: {}", error);
            let grads1 = vec![error];
            let grads2 = layer2.backwards(&grads1);
            layer1.backwards(&grads2);
        }
        let outp1 = layer1.forward(&input);
        let outp2 = layer2.forward(&outp1);
        let error_delta = outp2[0] - target;
        assert!(error_delta < 0.002);
    }
}
