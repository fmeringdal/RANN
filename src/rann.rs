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
            layers.push(LayerDense::new(input_size.clone(), output_size.clone()));
        }

        Self { layers }
    }

    pub fn forward(&mut self, input: &Vec<f32>) -> Vec<f32> {
        let mut output = input.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
            println!("New outp dim: {:?}", output)
        }
        output
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn rann_dims() {
        let output_size = 10;
        let mut rann = Rann::new(&vec![5, 4, 3, 7, 9, output_size]);
        let input = vec![-1., 3., 5., 3., -6.];
        let output = rann.forward(&input);
        println!("Output: {:?}", output);
        assert_eq!(output.len(), output_size);
    }
}
