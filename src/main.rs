use serde::Deserialize;

#[derive(Deserialize)]
struct Weights{
    hidden_weights: Vec<Vec<f64>>,
    hidden_bias: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>
}

fn hidden_layer(input: &Vec<f64>, weights: &Vec<Vec<f64>>, bias: &Vec<f64>) -> Vec<f64>{
    let mut outputs: Vec<f64> = Vec::new();
    for i in 0..weights.len(){
        let mut weighted_sum = bias[i];
        for j in 0..input.len(){
            weighted_sum += weights[i][j] * input[j];
        }
        let activated = weighted_sum.max(0.0);
        outputs.push(activated);
    }
    outputs
}

fn main() {
    let weights_json = std::fs::read_to_string("weights.json")
    .expect("could not read file");

    let weights: Weights = serde_json::from_str(&weights_json)
    .expect("could not parse file");

    println!("{:?}", weights.hidden_bias);
} 
