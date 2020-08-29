use crate::activations::{ActivationFunc, Sigmoid, RELU};
use crate::math::{softmax, softmax_derivative};
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
        //println!("Weight update: {:?}", delta_weights);
        // panic!();
        for i in 0..delta_weights.len() {
            self.weights[i] -= delta_weights[i];
        }
    }
}

pub trait Layer {
    fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32>;
    fn backwards(&mut self, inputs: &Vec<f32>) -> Vec<f32>;
}

pub struct LayerDense {
    nodes: Vec<Node>,
    input_nodes_count: usize,
    nodes_count: usize,
    pub outputs: Vec<f32>,
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
            outputs: vec![],
        }
    }
}

impl Layer for LayerDense {
    fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let outputs = self
            .nodes
            .iter_mut()
            .map(|node| node.forward(inputs))
            .collect();
        self.outputs = outputs;
        self.outputs.clone()
    }

    fn backwards(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
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

pub struct LayerSoftmax {
    input_nodes_count: usize,
    inputs: Vec<f32>,
    pub outputs: Vec<f32>,
}

impl LayerSoftmax {
    pub fn new(input_nodes_count: usize) -> Self {
        Self {
            input_nodes_count,
            inputs: vec![],
            outputs: vec![],
        }
    }
}

impl Layer for LayerSoftmax {
    fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        self.inputs = inputs.clone();
        self.outputs = softmax(inputs);
        self.outputs.clone()
    }

    fn backwards(&mut self, derivatives: &Vec<f32>) -> Vec<f32> {
        softmax_derivative(&self.inputs)
            .iter()
            .zip(derivatives)
            .map(|(sd, d)| {
                if sd.is_nan() {
                    return 0.1 * d;
                }
                sd * d
            })
            .collect()
    }
}

pub struct RannV2 {
    layers: Vec<LayerDense>,
    //output_layer: LayerSoftmax,
}

impl RannV2 {
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
        // self.output_layer.forward(&output)
        output
    }

    pub fn backwards(&mut self, target: &Vec<f32>) {
        let mut error_grad: Vec<f32> = target
            .iter()
            .zip(self.layers[self.layers.len() - 1].outputs.clone())
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

    #[test]
    fn rannv2_test() {
        let mut rannv2 = RannV2::new(vec![3, 3, 8]);
        let result = rannv2.forward(&vec![3., 4., 7.]);
        assert!(result.into_iter().sum::<f32>() == 1.);
    }

    #[test]
    fn rannv2_2test() {
        let mut rannv2 = RannV2::new(vec![30, 20, 10]);
        for _ in 0..100000 {
            let result = rannv2.forward(&vec![
                0.3, 0.4, 0.5, 0.1, 0.6, 0.7, 0.11, -0.5, -0.8, 0.17, 0.3, 0.4, 0.5, 0.1, 0.6, 0.7,
                0.11, -0.5, -0.8, 0.17, 0.3, 0.4, 0.5, 0.1, 0.6, 0.7, 0.11, -0.5, -0.8, 0.17,
            ]);
            println!("Pred: {:?}", result);
            rannv2.backwards(&vec![0., 1., 0., 0., 0., 0., 0., 0., 0., 0.]);
        }
    }

    #[test]
    fn rannv2_3test() {
        let mut rannv2 = RannV2::new(vec![3, 2, 2]);
        let train_set_1 = vec![vec![vec![1., 1., 1.], vec![1., 0.]]; 5000];
        let train_set_2 = vec![vec![vec![5., 5., 5.], vec![0., 1.]]; 5000];

        for _ in 0..50000 {
            for i in 0..10000 {
                let (input, target) = match i % 2 {
                    0 => (train_set_1[i / 2][0].clone(), train_set_1[i / 2][1].clone()),
                    _ => (
                        train_set_2[(i - 1) / 2][0].clone(),
                        train_set_2[(i - 1) / 2][1].clone(),
                    ),
                };
                let result = rannv2.forward(&input);
                println!("Result: {:?}", result);
                println!("Target: {:?}", target);
                if (result[0] > result[1] && target[0] > target[1])
                    || (result[0] < result[1] && target[0] < target[1])
                {
                    println!("Correct");
                } else {
                    println!("False");
                }
                rannv2.backwards(&target);
            }
        }
    }

    #[test]
    fn xor() {
        let mut rannv2 = RannV2::new(vec![2, 10, 1]);
        let train_set = vec![
            (vec![0., 0.], vec![0.]),
            (vec![0., 1.], vec![1.]),
            (vec![1., 0.], vec![1.]),
            (vec![1., 1.], vec![0.]),
        ];

        for _ in 0..5000000 {
            for i in 0..4 {
                let (input, target) = train_set[i].clone();
                let result = rannv2.forward(&input);
                println!("Result: {:?}", result);
                println!("Target: {:?}", target);
                let result = result[0];
                let target2 = target[0];
                println!(
                    "Correct: {}",
                    (result > 0.5 && target2 > 0.5) || (result < 0.5 && target2 < 0.5)
                );
                rannv2.backwards(&target);
            }
        }
    }
}
