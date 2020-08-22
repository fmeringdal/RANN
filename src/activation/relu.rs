pub struct RELU {
    output: Option<Vec<i64>>,
}

impl RELU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: &Vec<i64>) -> Vec<i64> {
        let output: Vec<i64> = inputs
            .iter()
            .map(|x| {
                if *x > 0 {
                    return *x;
                }
                0
            })
            .collect();

        let output2 = output.clone();
        self.output = Some(output);

        output2
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
