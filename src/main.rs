use serde::Deserialize;

#[derive(Deserialize)]
struct Weights{
    hidden_weights: Vec<Vec<f64>>,
    hidden_bias: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>
}

fn hidden_layer(input: &Vec<f64>, weights: &Vec<Vec<f64>>, bias: &Vec<f64>, apply_relu: bool) -> Vec<f64>{
    let mut outputs: Vec<f64> = Vec::new();
    for i in 0..weights.len(){
        let mut weighted_sum = bias[i];
        for j in 0..input.len(){
            weighted_sum += weights[i][j] * input[j];
        }
        if apply_relu{
            weighted_sum = weighted_sum.max(0.0);
        }
        outputs.push(weighted_sum);
    }
    outputs
}

fn main() {
    let weights_json = std::fs::read_to_string("weights.json")
    .expect("could not read file");

    let weights: Weights = serde_json::from_str(&weights_json)
    .expect("could not parse file");

    let inputs_test = vec![4.900000095367432, 3.0, 1.399999976158142, 0.20000000298023224];
    let hidden_test = hidden_layer(&inputs_test, &weights.hidden_weights, &weights.hidden_bias, true);
    let logits = hidden_layer(&hidden_test, &weights.output_weights, &weights.output_bias, false);
    println!("{:?}", logits);
} 
