# rust-inference

A neural network inference engine written from scratch in Rust. PyTorch trains the
model, Rust runs the forward pass. No ML libraries on the Rust side.

## What it does

A small MLP is trained on the Iris dataset in PyTorch, and its learned weights and
biases are exported to JSON. The Rust program loads that JSON and reimplements the
forward pass by hand: matrix multiplication, bias addition, ReLU, and argmax, all
written out as explicit loops.

The point is the split. Training needs autograd, an optimizer, and a lot of
machinery. Inference needs none of that. It is arithmetic over a fixed set of
numbers, and it can run anywhere those numbers can be read.

## Architecture

4 inputs -> 8 hidden units (ReLU) -> 3 output classes

The output layer produces raw logits. Softmax is skipped because argmax over logits
gives the same class as argmax over probabilities.

## Files

- `train.py`: trains the model, writes `weights.json` and `test_samples.json`
- `src/main.rs`: loads both files, runs inference, compares against PyTorch
- `weights.json`: the trained parameters
- `test_samples.json`: all 150 inputs and the predictions PyTorch produced for them

## Running it

```
python train.py
cargo run
```

The Python step must come first, since it writes both JSON files. Rerunning it
retrains from a new random initialization and rewrites both, which keeps them
consistent with each other.

## Results

150 of 150 predictions match PyTorch exactly.

That number is a claim about implementation agreement, not about model accuracy.
It means the Rust arithmetic reproduces PyTorch's arithmetic on every sample, which
is what an inference engine is supposed to do.

## Limitations

- **No train/test split.** The model is trained on all 150 rows and compared on the
  same 150, so nothing here says whether it generalizes or memorized.
- **Predictions, not labels.** `test_samples.json` stores what PyTorch predicted,
  not the true species, which is deliberate: the comparison is Rust against PyTorch.
- **Fixed architecture.** The layer function generalizes over sizes, but `main`
  assumes exactly two layers with ReLU on the first.
- **Naive loops.** No SIMD, no BLAS, no batching. Clarity over speed.
- **f32 to f64.** PyTorch trains in float32, and the exported values widen to float64
  in Rust. Harmless here because the logit gaps are wide, but a less confident model
  could see argmax flips from rounding.

## Why

Written to learn Rust and to understand what a trained model actually is once the
training framework is gone: a few dozen floats and some loops.
