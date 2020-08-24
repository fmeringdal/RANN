use crate::layers::dense::LayerDense;

pub struct Rann {
    layers: Vec<LayerDense>,
}

impl Rann {
    pub fn new(layer_sizes: &Vec<usize>) -> Self {
        let mut layers = Vec::new();
        for i in 0..layer_sizes.len() - 1 {
            let input_size = layer_sizes[i];
            let output_size = layer_sizes[i + 1];
            layers.push(LayerDense::new(
                input_size.clone(),
                output_size.clone(),
                i < layer_sizes.len() - 2,
            ));
        }

        Self { layers }
    }

    pub fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
            println!("After forward from layer: {:?}", output);
        }
        output
    }

    fn backward(&mut self, desired_output: &Vec<f32>) {
        let mut desired_output = desired_output.clone();
        for layer in self.layers.iter_mut().rev() {
            desired_output = layer.backward(&desired_output);
        }
    }

    pub fn train(&mut self, training_set: &Vec<Vec<f32>>, output_set: &Vec<Vec<f32>>) {
        // calculate cost function
        // calculate derivative cost by each node in output
        // pass back

        for (training, output) in training_set.iter().zip(output_set) {
            let pred_output = self.forward(&training);
            let cost: f32 = pred_output
                .iter()
                .zip(output)
                .map(|(p, t)| (p - t).powi(2))
                .sum();
            println!("Cost: {}", cost);
            let grad = pred_output
                .iter()
                .zip(output)
                .map(|(p, t)| 2. * (p - t))
                .collect();
            self.backward(&grad);
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rann_dims() {
        let output_size = 1;
        let mut rann = Rann::new(&vec![5, 4, 3, 7, 9, output_size]);
        let input = vec![-1., 3., 5., 3., -6.];
        let output = rann.forward(&input);
        println!("Output: {:?}", output);
        rann.backward(&vec![1.]);
        assert_eq!(output.len(), output_size);
    }

    #[test]
    fn train_rann() {
        let training = vec![vec![4., -2., 34., -6., 2.]; 20000];
        let output = vec![vec![0.17]; 20000];
        let mut rann = Rann::new(&vec![5, 4, 2, 1]);
        rann.train(&training, &output);
        let output = rann.forward(&training[0]);
        println!("Got last output: {:?}", output);
    }
}
