use crate::math::{dot, mat_multiply_mat, matrix_multiply_vec};
use rand::Rng;

struct Network {
    num_layers: usize,
    sizes: Vec<usize>,
    weights: Vec<Vec<Vec<f32>>>,
    biases: Vec<Vec<f32>>,
}

impl Network {
    pub fn new(sizes: Vec<usize>) -> Self {
        let num_layers = sizes.len();
        let sizes_cp = sizes.clone();

        let mut rng = rand::thread_rng();

        let mut t = vec![vec![vec![0.]]];
        t = sizes_cp[..num_layers - 1]
            .iter()
            .zip(sizes_cp[1..].iter())
            .map(|(x, y)| vec![vec![rng.gen(); *x]; *y])
            .collect();

        println!("Hello {:?}", t);

        Self {
            sizes,
            num_layers,
            biases: sizes_cp[1..]
                .iter()
                .map(|size| vec![0; *size].iter().map(|_| rng.gen()).collect())
                .collect(),
            weights: sizes_cp[..num_layers - 1]
                .iter()
                .zip(sizes_cp[1..].iter())
                .map(|(x, y)| {
                    vec![vec![0; *x]; *y]
                        .iter()
                        .map(|vec| vec.iter().map(|_| rng.gen()).collect())
                        .collect()
                })
                .collect(),
        }
    }

    pub fn feedforward(&self, input: &Vec<f32>) -> Vec<f32> {
        let mut a = input.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            a = matrix_multiply_vec(&w, &a)
                .iter()
                .zip(b)
                .map(|(o, b1)| sigmoid(o + b1))
                .collect();
        }
        a
    }

    pub fn SGD(
        &mut self,
        training_data: &Vec<(Vec<f32>, Vec<f32>)>,
        epochs: usize,
        mini_batch_size: usize,
        eta: f32,
    ) {
        let n = training_data.len();
        for _ in 0..epochs {
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
        let mut nabla_b = self.biases.clone();
        let mut nabla_w = self.weights.clone();

        println!("INit {:?}  ", nabla_w);

        for (x, y) in mini_batch.into_iter() {
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x.clone(), y);
            nabla_b = nabla_b
                .iter()
                .zip(delta_nabla_b)
                .map(|(nb, dnb)| nb.iter().zip(dnb).map(|(val1, val2)| val1 + val2).collect())
                .collect();
            nabla_w = nabla_w
                .iter()
                .zip(delta_nabla_w)
                .map(|(nw, dnw)| nw.iter().zip(dnw).map(|(val1, val2)| val2).collect())
                .collect();
        }

        println!("Going to update weights: {:?}", nabla_w);
        for l in 0..self.weights.len() {
            self.weights[l] = self.weights[l]
                .iter()
                .zip(&nabla_w[l])
                .map(|(arr1, arr2)| self.sub_vecs(&arr1, &arr2, eta))
                .collect();
        }

        for l in 0..self.biases.len() {
            self.biases[l] = self.sub_vecs(&self.biases[l], &nabla_b[l], eta);
        }
    }

    fn sub_vecs(&self, vec1: &Vec<f32>, vec2: &Vec<f32>, eta: f32) -> Vec<f32> {
        vec1.iter()
            .zip(vec2)
            .map(|(val1, val2)| val1 - val2 * eta)
            .collect()
    }

    fn backprop(&mut self, x: Vec<f32>, y: &Vec<f32>) -> (Vec<Vec<f32>>, Vec<Vec<Vec<f32>>>) {
        let mut nabla_b = self.biases.clone();
        let mut nabla_w = self.weights.clone();

        let mut activations = vec![x];
        let mut zs = vec![];

        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            let mut z = matrix_multiply_vec(&w, &activations[activations.len() - 1]);
            let mut activation = vec![];
            for i in 0..z.len() {
                z[i] = z[i] + b[i];
                activation.push(sigmoid(z[i]));
            }
            zs.push(z);
            activations.push(activation);
        }

        let mut sigmoid_primes = vec![];
        for val in zs[zs.len() - 1].iter() {
            sigmoid_primes.push(sigmoid_prime(*val));
        }

        let delta = self
            .cost_derivative(&activations[activations.len() - 1], &y)
            .iter()
            .zip(sigmoid_primes)
            .map(|(x1, x2)| x1 * x2)
            .collect();

        nabla_b[self.biases.len() - 1] = delta;
        nabla_w[self.weights.len() - 1] = mat_multiply_mat(
            &vec![nabla_b[self.biases.len() - 1].clone()],
            &activations[activations.len() - 2..].to_vec(),
        );

        for l in 2..self.num_layers {
            let z = zs[zs.len() - l].clone();
            let sp = z.into_iter().map(|val| sigmoid_prime(val));
            let delta = mat_multiply_mat(
                &&self.weights[self.weights.len() - l + 1],
                &vec![nabla_b[self.biases.len() - 1].clone()],
            )
            .iter()
            .zip(sp)
            .map(|(t, l)| {
                let mut total = 0.;
                for s in t.iter() {
                    total = s * l;
                }
                total
            })
            .collect();
            nabla_b[self.biases.len() - l] = delta;
            nabla_w[self.weights.len() - l] = mat_multiply_mat(
                &vec![nabla_b[self.biases.len() - l].clone()],
                &vec![activations[activations.len() - l - 1].clone()],
            );
        }

        println!("nabla b: {:?}", nabla_b);
        println!("nabla w: {:?}", nabla_w);
        panic!();

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

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn do_int_test() {
        let mut n = Network::new(vec![2, 5, 1]);
        let training_data = vec![
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
            (vec![1., 1.], vec![2.]),
        ];
        n.SGD(&training_data, 50000, 2, 0.01);
        let pred = n.feedforward(&vec![1., 1.]);
        println!("Pred: {:?}", pred);
    }
}
