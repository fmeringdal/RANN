use crate::math::dot;
use rand::Rng;

pub struct LayerDense {
    weights: Vec<Vec<i64>>,
    biases: Vec<i64>,
    output: Option<Vec<i64>>,
}

fn gen_weights(count: usize) -> Vec<i64> {
    let mut rng = rand::thread_rng();
    vec![0; count]
        .iter()
        .map(|_| rng.gen_range(-10, 10))
        .collect()
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
        }
    }

    pub fn forward(&mut self, inputs: &Vec<i64>) -> Vec<i64> {
        let output: Vec<i64> = self
            .weights
            .iter()
            .map(|cell_weight| dot(cell_weight, inputs))
            .zip(&self.biases)
            .map(|(x, y)| x + y)
            .collect();

        let output2 = output.clone();
        self.output = Some(output);

        output2
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dense_single_layer() {
        let mut layer = LayerDense::new(3, 7);
        let output = layer.forward(&vec![1, 1, 1]);
        assert_eq!(output.len(), 7);
    }

    #[test]
    fn dense_multi_layer() {
        let mut layer1 = LayerDense::new(5, 4);
        let mut layer2 = LayerDense::new(4, 3);
        let mut layer3 = LayerDense::new(3, 7);
        let input = &vec![1, -3, 3, 1, 6];
        let output = layer1.forward(input);
        assert_eq!(output.len(), 4);
        let output = layer2.forward(&output);
        assert_eq!(output.len(), 3);
        let output = layer3.forward(&output);
        assert_eq!(output.len(), 7);
    }
}
