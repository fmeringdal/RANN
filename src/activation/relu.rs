trait ActivationFunc {
    fn compute(val: f32) -> f32;
    fn compute_derivative(val: f32) -> f32;
}

struct RELU {}
struct Sigmoid {}

impl ActivationFunc for RELU {
    fn compute(val: f32) -> f32 {
        if val > 0. {
            return val;
        }
        0.
    }

    fn compute_derivative(val: f32) -> f32 {
        if val > 0. {
            return 1.;
        }
        0.
    }
}

impl ActivationFunc for Sigmoid {
    fn compute(val: f32) -> f32 {
        1_f32 / (1_f32 + 2.71828_f32.powf(-val))
    }

    fn compute_derivative(val: f32) -> f32 {
        val * (1. - val)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn relu_compute() {
        assert_eq!(RELU::compute(-1.), 0.);
        assert_eq!(RELU::compute(0.), 0.);
        assert_eq!(RELU::compute(1.2), 1.2);
    }

    #[test]
    fn relu_compute_derivative() {
        assert_eq!(RELU::compute_derivative(-1.), 0.);
        assert_eq!(RELU::compute_derivative(0.), 0.);
        assert_eq!(RELU::compute_derivative(1.3), 1.);
    }

    #[test]
    fn sigmoid_compute() {
        assert!(Sigmoid::compute(-100.) < 0.01);
        println!("Val: {}", Sigmoid::compute(0.2));
        assert!(Sigmoid::compute(-0.2) > 0.45 && 0.451 > Sigmoid::compute(-0.2));
        assert!(Sigmoid::compute(0.) == 0.5);
        assert!(Sigmoid::compute(0.2) > 0.549 && Sigmoid::compute(0.2) < 0.5499);
        assert!(Sigmoid::compute(100.) > 0.99);
    }

    #[test]
    fn sigmoid_compute_derivative() {
        assert_eq!(Sigmoid::compute_derivative(-1.), -2.);
        assert_eq!(Sigmoid::compute_derivative(0.), 0.);
        assert_eq!(Sigmoid::compute_derivative(1.), 0.);
        assert_eq!(Sigmoid::compute_derivative(2.), -2.);
    }
}
