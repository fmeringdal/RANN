//tutorial-read-01.rs
use crate::activations::{Sigmoid, RELU};
use crate::math::one_hot_encode;
use crate::node::LayerDense;
use crate::rann::Rann;
use std::env;
use std::error::Error;
use std::ffi::OsString;
use std::fs::File;
use std::process;
use std::rc::Rc;

fn run() -> Result<(Vec<Vec<f32>>, Vec<Vec<f32>>), Box<dyn Error>> {
    // let file_path = get_first_arg()?;
    let file_path = "../../Downloads/mnist_train.csv";
    println!("Path: {:?}", file_path);
    let file = File::open(file_path)?;
    let mut rdr = csv::Reader::from_reader(file);

    let mut training_set: Vec<Vec<f32>> = Vec::new();
    let mut labels: Vec<Vec<f32>> = Vec::new();

    let max = 1;
    let mut counter = 0;
    for result in rdr.records() {
        if counter > max {
            break;
        }
        let mut record = result?;
        //        let val = record.(",").collect();
        let label = record.get(0).unwrap().parse::<usize>().unwrap();
        let mut training = vec![0.0_f32; 28 * 28];
        for i in 0..28 * 28 {
            training[i] = record.get(i + 1).unwrap().parse::<f32>().unwrap() / 255.;
        }
        labels.push(
            one_hot_encode(label, 10)
                .into_iter()
                .map(|val| val as f32)
                .collect(),
        );
        //labels.push(vec![label]);
        training_set.push(training);
        counter += 1;
    }

    Ok((training_set, labels))
}

/// Returns the first positional argument sent to this process. If there are no
/// positional arguments, then this returns an error.
fn get_first_arg() -> Result<OsString, Box<dyn Error>> {
    match env::args_os().nth(1) {
        None => Err(From::from("expected 1 argument, but got none")),
        Some(file_path) => Ok(file_path),
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn woll() {
        println!("Start");
        match run() {
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
            Ok((training_set, labels)) => {
                println!("Labes: {:?}", labels);
                let mut rann = Rann::new(&vec![28 * 28, 100, 1]);
                for _ in 0..100 {
                    rann.train(&training_set, &labels);
                }
                println!("Prediction for 10 first");
                for i in 0..10 {
                    let pred = rann.forward(&training_set[i]);
                    println!("Predicted: {:?} vs real: {:?}", pred, labels[i]);
                }
            }
        }
    }

    #[test]
    fn mnist_layer() {
        let mut layer1 = LayerDense::new(28 * 28, 100, Rc::new(Sigmoid::new()));
        let mut layer2 = LayerDense::new(100, 10, Rc::new(Sigmoid::new()));

        println!("Start");
        match run() {
            Err(err) => {
                println!("{}", err);
                process::exit(1);
            }
            Ok((training_set, labels)) => {
                for _ in 0..1000 {
                    for (train_set, target) in training_set.iter().zip(labels.clone()) {
                        let out1 = layer1.forward(&train_set);
                        let out2 = layer2.forward(&out1);
                        let mut error = vec![0.; 10];
                        for i in 0..10 {
                            error[i] = out2[i] - target[i];
                        }
                        println!("-------------------------------------------");
                        println!("Target: {:?}", target);
                        println!("Pred: {:?}", out2);
                        println!("Error: {}", error.iter().sum::<f32>());
                        let grad3 = layer2.backwards(&error);
                        let grad2 = layer1.backwards(&grad3);
                        layer1.backwards(&grad2);
                    }
                }
                println!("Prediction for 10 first");
            }
        }
    }
}
