use crate::math::dot;

pub struct LayerDense {
    weights: Vec<Vec<i64>>,
    biases: Vec<i64>,
}

impl LayerDense {
    pub fn new() -> Self {
        Self {
            weights: vec![vec![1, 2, 3], vec![3, 2, 1]],
            biases: vec![1, 2, 3],
        }
    }

    pub fn forward(&self, inputs: &Vec<i64>) -> Vec<i64> {
        self.weights
            .iter()
            .map(|cell_weight| dot(cell_weight, inputs))
            .zip(&self.biases)
            .map(|(x, y)| x + y)
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn dense_layers() {
        let layer = LayerDense::new();
        assert_eq!(layer.forward(&vec![1, 1, 1]), vec![7, 8])
    }
}
