use crate::activations::{ActivationFunc, Sigmoid, RELU};
use rand::Rng;

pub struct Node {
    pub inputs: Vec<f32>,
    pub output: f32,
    pub weights: Vec<f32>,
    pub activation: Box<dyn ActivationFunc>,
    pub number_of_inputs: usize,
}

impl Node {
    pub fn new(
        number_of_inputs: usize,
        activation: Box<dyn ActivationFunc>,
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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn node_forward() {
        let mut node = Node::new(3, Box::new(RELU::new()), Some(vec![1., 2., 3.]));
        let output = node.forward(&vec![1., 2., 3.]);
        assert_eq!(output, 14.);
    }

    #[test]
    fn node_derivatives_perfect_weights() {
        let mut node = Node::new(3, Box::new(RELU::new()), Some(vec![1., 2., 3.]));
        let output = node.forward(&vec![1., 2., 3.]);
        let dervs = node.compute_derivatives(0.);
        assert_eq!(dervs, vec![0., 0., 0.]);
    }

    #[test]
    fn node_derivatives() {
        let mut node = Node::new(3, Box::new(Sigmoid::new()), Some(vec![0.1, 0.3, 0.5]));
        let output = node.forward(&vec![1., 4., 5.]);
        let dervs = node.compute_derivatives(0.1502);
        assert_eq!(dervs, vec![0.002, 0.0079, 0.0099]);
    }

    #[test]
    fn node_layer() {
        let mut node1 = Node::new(3, Box::new(Sigmoid::new()), None);
        let mut node2 = Node::new(3, Box::new(Sigmoid::new()), None);
        let mut node3 = Node::new(2, Box::new(Sigmoid::new()), None);
        for _ in 0..100000 {
            let input = vec![1.1, 1.2, 1.3];
            let output1 = node1.forward(&input);
            let output2 = node2.forward(&input);
            let output3 = node3.forward(&vec![output1, output2]);
            println!("Out: {}", output3);
            println!("Error: {}", (output3 - 0.25).powi(2));
            let error_delta = output3 - 0.25;
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
            // println!("NOde 1 de: {:?}", node1de);
            // println!("NOde 2 de: {:?}", node2de);
            // println!("NOde 3 de: {:?}", node3de);
        }
    }
}
