pub struct RELU {
    output: Option<Vec<i64>>,
}

impl RELU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: &Vec<i64>) -> Vec<i64> {
        let output = inputs
            .iter()
            .map(|x| {
                if *x > 0 {
                    return *x;
                }
                0
            })
            .collect();

        output
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn relu() {
        let mut r = RELU::new();
        assert_eq!(r.forward(&vec![-1, 0, 1]), [0, 0, 1])
    }
}
