use crate::math::dot;
use rand::Rng;

pub struct LayerDense {
    weights: Vec<Vec<f32>>,
    biases: Vec<f32>,
    output: Option<Vec<f32>>,
    input: Option<Vec<f32>>,
    input_nodes_count: usize,
    output_nodes_count: usize,
}

fn gen_weights(count: usize) -> Vec<f32> {
    let mut rng = rand::thread_rng();
    vec![0; count].iter().map(|_| rng.gen::<f32>()).collect()
}

impl LayerDense {
    pub fn new(input_nodes: usize, output_nodes: usize) -> Self {
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

    pub fn backward(&mut self, desired_output: &Vec<f32>) -> Vec<f32> {
        let desired_input: Vec<f32> = Vec::with_capacity(self.input_nodes_count);
        let error: Vec<f32> = desired_output
            .iter()
            .zip(self.output.as_ref().unwrap())
            .map(|(desired, actual)| actual - desired)
            .collect();

        self.weights = self
            .weights
            .iter()
            .zip(error)
            .map(|(node_weights, error)| node_weights.iter().map(|weight| weight * error).collect())
            .collect();

        Vec::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dense_single_layer() {
        let mut layer = LayerDense::new(3, 7);
        let output = layer.forward(&vec![1.0, 1.0, 1.0]);
        assert_eq!(output.len(), 7);
    }

    #[test]
    fn dense_multi_layer() {
        let mut layer1 = LayerDense::new(5, 4);
        let mut layer2 = LayerDense::new(4, 3);
        let mut layer3 = LayerDense::new(3, 7);
        let input = &vec![1.0, -3.0, 3.0, 1.0, 6.0];
        let output = layer1.forward(input);
        assert_eq!(output.len(), 4);
        let output = layer2.forward(&output);
        assert_eq!(output.len(), 3);
        let output = layer3.forward(&output);
        assert_eq!(output.len(), 7);
    }
}
