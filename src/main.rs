use serde::Deserialize;

#[derive(Deserialize)]
struct Weights{
    hidden_weights: Vec<Vec<f64>>,
    hidden_bias: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>
}

fn main() {
    let weights_json = std::fs::read_to_string("weights.json")
    .expect("could not read file");

    let weights: Weights = serde_json::from_str(&weights_json)
    .expect("could not parse file");

    println!("{:?}", weights.hidden_bias);
} 
