use crate::activations::{ActivationFunc, Sigmoid, RELU};
use crate::math::{softmax, softmax_derivative};
use rand::distributions::{Distribution, Normal};
use rand::Rng;
use std::rc::Rc;

pub struct Node {
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
            number_of_inputs,
            activation,
        }
    }

    fn gen_weights(count: usize) -> Vec<f32> {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(0., 1.);
        vec![0; count]
            .iter()
            .map(|_| normal.sample(&mut rng) as f32)
            .collect()
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> f32 {
        let output = self.weights.iter().zip(inputs).map(|(w, i)| w * i).sum();
        if self.number_of_inputs == 100 && false {
            println!("Inputs to node: {:?}", inputs);
            println!("And outputs before acts: {:?}", output);
        }
        let output = self.activation.compute(output);
        self.output = output;
        output
    }

    pub fn compute_derivatives(&self, inputs: &Vec<f32>, dc_da: f32) -> Vec<f32> {
        let part1_update = dc_da * self.activation.compute_derivative(self.output);
        let mut weight_derivatives = vec![0.; self.number_of_inputs];
        for i in 0..self.number_of_inputs {
            weight_derivatives[i] = part1_update * inputs[i];
        }
        weight_derivatives
    }

    pub fn update_weights(&mut self, delta_weights: Vec<f32>) {
        //println!("Weight update: {:?}", delta_weights);
        // panic!();
        for i in 0..delta_weights.len() {
            self.weights[i] -= delta_weights[i];
        }
    }
}
