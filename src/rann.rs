use crate::activations::{ActivationFunc, Sigmoid, RELU};
use crate::layers::dense::{Layer, LayerDense};
use std::rc::Rc;

pub struct Rann {
    layers: Vec<LayerDense>,
    //output_layer: LayerSoftmax,
}

impl Rann {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        assert!(layer_sizes.len() > 2);

        let mut layers = vec![];
        for i in 1..layer_sizes.len() {
            layers.push(LayerDense::new(
                layer_sizes[i - 1],
                layer_sizes[i],
                Rc::new(Sigmoid::new()),
            ));
        }

        Self {
            layers,
            //output_layer: LayerSoftmax::new(layer_sizes[layer_sizes.len() - 1]),
        }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let mut output = inputs.clone();
        for layer in self.layers.iter_mut() {
            output = layer.forward(&output);
        }
        output
    }

    pub fn backwards(&mut self, target: &Vec<f32>) {
        let mut error_grad: Vec<f32> = target
            .iter()
            .zip(self.layers[self.layers.len() - 1].outputs())
            .map(|(t, pred)| pred - t)
            .collect();
        let debug = false;
        if debug {
            println!("Error: {:?}", error_grad);
        }
        // let mut error_grad = self.output_layer.backwards(&error_grad);
        if debug {
            println!("Error: {:?}", error_grad);
        }
        for layer in self.layers.iter_mut().rev() {
            error_grad = layer.backwards(&error_grad);
            if debug {
                println!("Error: {:?}", error_grad);
            }
        }
    }
}
