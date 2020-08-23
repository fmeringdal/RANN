use crate::math::{dot, matrix_multiply_vec};

struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        Self {
            sizes,
            num_layers: sizes.len(),
            biases: sizes[1..].iter().map(|size| vec![0.1; *size]).collect(),
            weights: sizes[..sizes.len() - 1]
                .iter()
                .zip(sizes[1..].iter())
                .map(|(x, y)| vec![vec![0.1; *x]; *y])
                .collect(),
        }
    }

    pub fn SGD(
        &mut self,
        training_data: &Vec<(Vec<f32>, Vec<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
    ) {
        let n = training_data.len();
        for j in 0..epochs {
            let mut mini_batches = vec![];
            for k in 0..(n / mini_batch_size) {
                mini_batches
                    .push(&training_data[(k * mini_batch_size)..((k + 1) * mini_batch_size)]);
            }
            for mini_batch in mini_batches.into_iter() {
                self.update_mini_batch(mini_batch, eta);
            }
        }
    }

    fn update_mini_batch(&mut self, mini_batch: &[(Vec<f32>, Vec<f32>)], eta: f32) {
        let mut nabla_b = vec![vec![0.; self.biases.len()]; 200];
        let mut nabla_w = vec![vec![0.; self.weights.len()]; 300];
        for (x, y) in mini_batch.iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, *y);
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b)
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = nabla_w
                .iter()
                .zip(delta_nabla_w)
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }
        self.weights = self
            .weights
            .iter()
            .zip(nabla_w)
            .map(|(w, nw)| self.sub_vecs(w, nw, eta))
            .collect();
        self.biases = self
            .biases
            .iter()
            .zip(nabla_b)
            .map(|(b, nb)| self.sub_vecs(b, nb, eta))
            .collect();
    }

    fn sub_vecs(&self, vec1: &Vec<f32>, vec2: &Vec<f32>, eta: f32) -> Vec<f32> {
        vec1.iter()
            .zip(vec2)
            .map(|(v1, v2)| v1 - eta * v2)
            .collect()
    }

    fn backprop(&mut self, x: &Vec<f32>, y: &Vec<f32>) -> (Vec<f32>, Vec<f32>) {
        let mut nabla_b = vec![vec![0, self.biases[0].len()]; self.biases.len()];
        let mut nabla_w = vec![
            vec![vec![0, self.weights[0][0].len()]; self.weights[0].len()];
            self.weights.len()
        ];

        let mut activation = x;
        let mut activations = vec![x];
        let mut zs = vec![];

        for (b, w) in self.biases.into_iter().zip(self.weights) {
            let mut z = matrix_multiply_vec(&w, activation);
            let mut activation = vec![];
            for i in 0..z.len() {
                z[i] = z[i] + b[i];
                activation.push(sigmoid(z[i]));
            }
            zs.push(z);
            activations.push(&activation);
        }

        let sigmoid_primes = zs[zs.len() - 1]
            .iter()
            .map(|val| sigmoid_prime(*val))
            .collect();

        let delta = self
            .cost_derivative(activations[activations.len() - 1], &y)
            .iter()
            .zip(sigmoid_primes)
            .map(|(x1, x2)| x1 * x2)
            .collect();

        nabla_b[nabla_b.len() - 1] = delta;
        nabla_w[nabla_w.len() - 1] = dot(delta, activations[activations.len() - 2..]);

        for l in 2..self.num_layers {
            let z = zs[l];
            let sp = sigmoid_prime(z);
            let delta = dot(self.weights[-l + 1], delta) * sp;
            nabla_b[-l] = delta;
            nabla_w[l] = dot(delta, activations[-l - 1]);
        }
        return (nabla_b.to_vec(), nabla_w.to_vec());
    }

    fn cost_derivative(&mut self, output_activations: &Vec<f32>, y: &Vec<f32>) -> Vec<f32> {
        output_activations
            .iter()
            .zip(y)
            .map(|(o, y)| o - y)
            .collect()
    }
}

pub fn sigmoid(z: f32) -> f32 {
    return 1_f32 / (1_f32 + 2.718_f32.powf(z));
}

pub fn sigmoid_prime(z: f32) -> f32 {
    let sig = sigmoid(z);
    sig * (1_f32 - sig)
}
