use std::cmp::Ordering;

pub fn dot(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x * y).sum()
}

pub fn matrix_multiply_vec(mat: &Vec<Vec<f32>>, vec: &Vec<f32>) -> Vec<f32> {
    let mut res = vec![];
    for row in mat.iter() {
        res.push(dot(row, vec));
    }
    res
}

pub fn mat_multiply_mat(vec: &Vec<Vec<f32>>, mat: &Vec<Vec<f32>>) -> Vec<Vec<f32>> {
    let mut res = vec![vec![0.; mat[0].len()]; vec.len()];
    for i in 0..res.len() {
        for j in 0..res[0].len() {
            let col_mat_2 = mat.iter().map(|row| row[j]);
            res[i][j] = vec[i]
                .iter()
                .zip(col_mat_2)
                .map(|(x, y)| x * y)
                .sum::<f32>();
        }
    }

    res
}

pub fn cut_val(val: f32, max: f32) -> f32 {
    if val > max {
        max
    } else if -max < val {
        -max
    } else {
        val
    }
}

pub fn one_hot_encode(i: usize, size: usize) -> Vec<usize> {
    let mut one_hot_encoded = vec![0; size];
    one_hot_encoded[i] = 1;
    one_hot_encoded
}

pub fn mean_squared_error(vec1: &Vec<f32>, vec2: &Vec<f32>) -> f32 {
    let size = vec1.len();
    if size == 0 {
        return 0.;
    }

    let error: f32 = vec1
        .iter()
        .zip(vec2.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum();

    error / size as f32
}

pub fn softmax(vals: &Vec<f32>) -> Vec<f32> {
    let e = 2.71828_f32;
    let max_val = vals
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal))
        .unwrap_or(&0.);
    let vals = vals.iter().map(|val| e.powf(*val - max_val));
    let denominator: f32 = vals.clone().sum();
    vals.map(|val| val / denominator).collect()
}

pub fn softmax_derivative(vals: &Vec<f32>) -> Vec<f32> {
    let e = 2.71828_f32;
    let vals: Vec<f32> = vals.iter().map(|val| e.powf(*val)).collect();
    let mut dervs = vec![0.; vals.len()];
    let total: f32 = vals.iter().sum();
    let total_square = total.powi(2);
    for i in 0..vals.len() {
        dervs[i] = vals[i] * (total - vals[i]) / total_square;
    }

    dervs
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dot_product() {
        assert_eq!(dot(&vec![1.0], &vec![1.0]), 1.0);
        assert_eq!(dot(&vec![1.0, 2.0], &vec![1.0, 2.0]), 5.0);
        assert_eq!(dot(&vec![], &vec![]), 0.);
        assert_eq!(dot(&vec![-1.], &vec![-1.]), 1.);
        assert_eq!(dot(&vec![-1., 1., 2.], &vec![-1., 1., 2.]), 6.);
    }

    #[test]
    fn mean_squared() {
        assert_eq!(mean_squared_error(&vec![], &vec![]), 0.);
        assert_eq!(mean_squared_error(&vec![1., -4.], &vec![1., -4.]), 0.);
        assert_eq!(mean_squared_error(&vec![1., -1.], &vec![0., -5.]), 8.5);
    }

    #[test]
    fn vec_mult_mat() {
        let res = mat_multiply_mat(&vec![vec![4.], vec![6.]], &vec![vec![-2., 5.]]);
        assert_eq!(res[0], vec![-8., 20.]);
        assert_eq!(res[1], vec![-12., 30.]);

        let res = mat_multiply_mat(
            &vec![vec![4., -2., 2.], vec![1., 5., -9.]],
            &vec![vec![12., -5.], vec![-3., -2.], vec![4., 2.]],
        );
        assert_eq!(res[0], vec![62., -12.]);
        assert_eq!(res[1], vec![-39., -33.]);
    }

    #[test]
    fn softmax_test() {
        let test1 = vec![1.];
        let res1 = softmax(&test1);
        assert_eq!(res1, vec![1.]);

        let test2 = vec![1., 1.];
        let res2 = softmax(&test2);
        assert_eq!(res2, vec![0.5, 0.5]);

        let test2 = vec![1., 1., 2.];
        let res2 = softmax(&test2);
        let expected = vec![0.2119, 0.2119, 0.5761];
        for (r, t) in res2.iter().zip(expected) {
            assert!(r - t < 0.01);
        }
    }

    #[test]
    fn softmax_der_test() {
        let test1 = vec![1.];
        let res1 = softmax_derivative(&test1);
        assert_eq!(res1, vec![0.]);

        let test2 = vec![1., 2.];
        let res2 = softmax_derivative(&test2);
        let expected = vec![0.19661, 0.19661];
        for (r, t) in res2.iter().zip(expected) {
            assert!(r - t < 0.01);
        }
    }
}
