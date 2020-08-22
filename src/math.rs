pub fn dot(vec1: &Vec<i64>, vec2: &Vec<i64>) -> i64 {
    vec1.iter().zip(vec2.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn dot_product() {
        assert_eq!(dot(&vec![1], &vec![1]), 1);
        assert_eq!(dot(&vec![1, 2], &vec![1, 2]), 5);
        assert_eq!(dot(&vec![], &vec![]), 0);
        assert_eq!(dot(&vec![-1], &vec![-1]), 1);
        assert_eq!(dot(&vec![-1, 1, 2], &vec![-1, 1, 2]), 6);
    }
}
