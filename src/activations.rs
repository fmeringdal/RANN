pub trait ActivationFunc {
    fn compute(&self, val: f32) -> f32;
    fn compute_derivative(&self, val: f32) -> f32;
}

pub struct RELU {}
pub struct Sigmoid {}

impl RELU {
    pub fn new() -> Self {
        Self {}
    }
}

impl Sigmoid {
    pub fn new() -> Self {
        Self {}
    }
}

impl ActivationFunc for RELU {
    fn compute(&self, val: f32) -> f32 {
        if val > 0. {
            return val;
        }
        0.
    }

    fn compute_derivative(&self, val: f32) -> f32 {
        if val > 0. {
            return 1.;
        }
        0.
    }
}

impl ActivationFunc for Sigmoid {
    fn compute(&self, val: f32) -> f32 {
        1_f32 / (1_f32 + 2.71828_f32.powf(-val))
    }

    fn compute_derivative(&self, val: f32) -> f32 {
        val * (1. - val)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn relu_compute() {
        let rel = RELU::new();
        assert_eq!(rel.compute(-1.), 0.);
        assert_eq!(rel.compute(0.), 0.);
        assert_eq!(rel.compute(1.2), 1.2);
    }

    #[test]
    fn relu_compute_derivative() {
        let rel = RELU::new();
        assert_eq!(rel.compute_derivative(-1.), 0.);
        assert_eq!(rel.compute_derivative(0.), 0.);
        assert_eq!(rel.compute_derivative(1.3), 1.);
    }

    #[test]
    fn sigmoid_compute() {
        let sig = Sigmoid::new();
        assert!(sig.compute(-100.) < 0.01);
        println!("Val: {}", sig.compute(0.2));
        assert!(sig.compute(-0.2) > 0.45 && 0.451 > sig.compute(-0.2));
        assert!(sig.compute(0.) == 0.5);
        assert!(sig.compute(0.2) > 0.549 && sig.compute(0.2) < 0.5499);
        assert!(sig.compute(100.) > 0.99);
    }

    #[test]
    fn sigmoid_compute_derivative() {
        let sig = Sigmoid::new();
        assert_eq!(sig.compute_derivative(-1.), -2.);
        assert_eq!(sig.compute_derivative(0.), 0.);
        assert_eq!(sig.compute_derivative(1.), 0.);
        assert_eq!(sig.compute_derivative(2.), -2.);
    }
}
