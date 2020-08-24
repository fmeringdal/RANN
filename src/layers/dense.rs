use crate::math::dot;
use rand::Rng;

pub struct LayerDense {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    output: Option<Vec<f32>>,
    input: Option<Vec<f32>>,
    input_nodes_count: usize,
    output_nodes_count: usize,
    activation: bool,
}

fn gen_weights(count: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    vec![0; count].iter().map(|_| rng.gen::<f32>()).collect()
}

impl LayerDense {
    pub fn new(input_nodes: usize, output_nodes: usize, activation: bool) -> Self {
        Self {
            weights: vec![0; output_nodes]
                .iter()
                .map(|_| gen_weights(input_nodes))
                .collect(),
            biases: gen_weights(output_nodes),
            output: None,
            input: None,
            input_nodes_count: input_nodes,
            output_nodes_count: output_nodes,
            activation,
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let output: Vec<f32> = self
            .weights
            .iter()
            .map(|cell_weight| dot(cell_weight, inputs))
            .zip(&self.biases)
            .map(|(x, y)| x + y)
            .map(|x| 1_f32 / (1_f32 + 2.718_f32.powf(x)))
            .collect();
        let output2 = output.clone();
        self.output = Some(output);
        self.input = Some(inputs.clone());

        output2
    }

    pub fn backward(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
        let learning_rate = 0.1;
        let outputs = self.output.as_ref().unwrap();
        let mut backward_derivatives = vec![0.;self.input_nodes_count];
        for i in 0..self.output_nodes_count {
            let output = outputs[i]; 
            let sigmoid_derivative = (1.-output)*output;
            self.weights[i] = self.weights[i].iter().map(|weight| {
                weight - learning_rate*sigmoid_derivative*derivatives[i]
            }).collect();
        }

        for i in 0..self.input_nodes_count {
            let mut der = 0.;
            for j in 0..self.output_nodes_count {
                let weight = self.weights[i][j];
                let out_deri = derivatives[j];
                let output = outputs[j];
                der += weight*output*out_deri;
            }
            backward_derivatives[i] = der;
        }

        backward_derivatives
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dense_single_layer() {
        let mut layer = LayerDense::new(3, 7, true);
        let output = layer.forward(&vec![1.0, 1.0, 1.0]);
        assert_eq!(output.len(), 7);
    }

    #[test]
    fn dense_multi_layer() {
        let mut layer1 = LayerDense::new(5, 4, true);
        let mut layer2 = LayerDense::new(4, 3, true);
        let mut layer3 = LayerDense::new(3, 7, true);
        let input = &vec![1.0, -3.0, 3.0, 1.0, 6.0];
        let output = layer1.forward(input);
        assert_eq!(output.len(), 4);
        let output = layer2.forward(&output);
        assert_eq!(output.len(), 3);
        let output = layer3.forward(&output);
        assert_eq!(output.len(), 7);
    }
}
