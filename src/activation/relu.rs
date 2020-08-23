pub struct RELU {
    output: Option<Vec<f32>>,
}

impl RELU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: &Vec<f32>) -> Vec<f32> {
        let output: Vec<f32> = inputs
            .iter()
            .map(|x| {
                if *x > 0. {
                    return *x;
                }
                0.
            })
            .collect();

        let output2 = output.clone();
        self.output = Some(output);

        output2
    }

    pub fn backward(&mut self, desired_output: &Vec<f32>) -> Vec<f32> {
        Vec::new()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn relu() {
        let mut r = RELU::new();
        assert_eq!(r.forward(&vec![-1., 0., 1.]), [0., 0., 1.])
    }
}
