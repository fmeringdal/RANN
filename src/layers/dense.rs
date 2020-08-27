use crate::activations::{ActivationFunc, Sigmoid, RELU};
use crate::math::{cut_val, dot};
use rand::Rng;

pub struct LayerDense {
    pub weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    output: Option<Vec<f32>>,
    input: Option<Vec<f32>>,
    input_nodes_count: usize,
    output_nodes_count: usize,
    activation: Box<dyn ActivationFunc>,
}

impl LayerDense {
    pub fn new(
        input_nodes: usize,
        output_nodes: usize,
        activation: Box<dyn ActivationFunc>,
    ) -> Self {
        Self {
            weights: vec![0; output_nodes]
                .iter()
                .map(|_| Self::gen_weights(input_nodes))
                .collect(),
            biases: vec![0.2; output_nodes],
            output: None,
            input: None,
            input_nodes_count: input_nodes,
            output_nodes_count: output_nodes,
            activation,
        }
    }

    fn gen_weights(count: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        vec![0; count].iter().map(|_| rng.gen::<f32>()).collect()
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let output: Vec<f32> = self
            .weights
            .iter()
            .map(|cell_weight| dot(cell_weight, inputs))
            .zip(&self.biases)
            .map(|(x, y)| self.activation.compute(x + y))
            .collect();
        let without_act: Vec<f32> = self
            .weights
            .iter()
            .map(|w| dot(w, inputs))
            .zip(&self.biases)
            .map(|(x, y)| x + y)
            .collect();
        let output2 = output.clone();
        self.output = Some(output);
        self.input = Some(inputs.clone());
        if self.output_nodes_count == 10 && false {
            println!("Biases: {:?}", self.biases);
            println!("Last output: {:?}", without_act);
        }
        output2
    }

    pub fn backward(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
        let learning_rate = 0.1;
        let outputs = self.output.as_ref().unwrap();
        let mut backward_derivatives = vec![0.; self.input_nodes_count];

        for i in 0..self.input_nodes_count {
            let mut der = 0.;
            for j in 0..self.output_nodes_count {
                let weight = self.weights[j][i];
                let out_deri = derivatives[j];
                let output = cut_val(outputs[j], 0.98);
                der += weight * self.activation.compute_derivative(output) * out_deri;
            }
            backward_derivatives[i] = der;
        }

        if self.output_nodes_count == 10 && false {
            println!("Backwards grads: {:?}", backward_derivatives);
        }

        let inputs = self.input.clone().unwrap();
        for i in 0..self.output_nodes_count {
            let output = cut_val(outputs[i], 0.98);
            for j in 0..self.weights[i].len() {
                //  let current_w = self.weights[i][j];
                let mut update = learning_rate
                    * self.activation.compute_derivative(output)
                    * inputs[j]
                    * derivatives[i];
                update = cut_val(update, 2.);
                self.weights[i][j] -= update;
                // println!("Weight update: {}", update);
            }
        }
        backward_derivatives
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dense_single_layer() {
        let mut layer = LayerDense::new(3, 7, Box::new(Sigmoid::new()));
        let output = layer.forward(&vec![1.0, 1.0, 1.0]);
        assert_eq!(output.len(), 7);
    }

    #[test]
    fn dense_multi_layer() {
        let mut layer1 = LayerDense::new(5, 4, Box::new(Sigmoid::new()));
        let mut layer2 = LayerDense::new(4, 3, Box::new(Sigmoid::new()));
        let mut layer3 = LayerDense::new(3, 7, Box::new(Sigmoid::new()));
        let input = &vec![1.0, -3.0, 3.0, 1.0, 6.0];
        let output = layer1.forward(input);
        assert_eq!(output.len(), 4);
        let output = layer2.forward(&output);
        assert_eq!(output.len(), 3);
        let output = layer3.forward(&output);
        assert_eq!(output.len(), 7);
    }
}
