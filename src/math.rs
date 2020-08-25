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
}
