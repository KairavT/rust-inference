use serde::Deserialize;

#[derive(Deserialize)]
struct TestSamples{
    inputs: Vec<Vec<f64>>,
    predictions: Vec<usize>
}

#[derive(Deserialize)]
struct Weights{
    hidden_weights: Vec<Vec<f64>>,
    hidden_bias: Vec<f64>,
    output_weights: Vec<Vec<f64>>,
    output_bias: Vec<f64>
}

fn predict(input: &Vec<f64>, weights: &Weights) -> usize{
    let hidden_test = hidden_layer(input, &weights.hidden_weights, &weights.hidden_bias, true);
    let logits = hidden_layer(&hidden_test, &weights.output_weights, &weights.output_bias, false);
    let argmax_result = argmax(&logits);
    argmax_result
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

fn argmax(logits: &Vec<f64>) -> usize{
    let mut best_index = 0;
    let mut best_value = logits[0];
    for i in 1..logits.len(){
        if logits[i] > best_value{
            best_index = i;
            best_value = logits[i];
        }
    }
    best_index
}

fn main() {
    let weights_json = std::fs::read_to_string("weights.json")
    .expect("could not read file");

    let weights: Weights = serde_json::from_str(&weights_json)
    .expect("could not parse file");

    let samples_json = std::fs::read_to_string("test_samples.json")
    .expect("could not read file");

    let samples: TestSamples = serde_json::from_str(&samples_json)
    .expect("could not parse file");

    for i in 0..samples.inputs.len(){
        let cur_input = &samples.inputs[i];
        let cur_prediction = predict(cur_input, &weights);
        let matching = cur_prediction == samples.predictions[i];
        println!("Prediction {}: {:?}, Sample Prediction: {:?}, Equal: {}",
                i, cur_prediction, samples.predictions[i], matching);
    }
    
} 
