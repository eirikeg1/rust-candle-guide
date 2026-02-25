# Candle Quick Reference

## What is Candle?

Candle is a minimalist ML framework for Rust by Hugging Face. It provides a PyTorch-like tensor API with CPU and GPU support, automatic differentiation, and neural network building blocks — all in pure Rust with no Python dependency.

## Setup

```toml
[dependencies]
candle-core = "0.9"
candle-nn = "0.9"
anyhow = "1"
```

Common imports:

```rust
use candle_core::{DType, Device, Module, Tensor, Var};
use candle_nn::{linear, loss, ops, Activation, Linear, Optimizer, VarBuilder, VarMap};
```

## Tensor Creation

```rust
let device = &Device::Cpu;  // or Device::cuda_if_available(0)?

// From data
let t = Tensor::new(&[1.0f32, 2.0, 3.0], device)?;             // 1D
let t = Tensor::new(&[[1.0f32, 2.0], [3.0, 4.0]], device)?;    // 2D

// Special tensors
let z = Tensor::zeros((3, 4), DType::F32, device)?;
let o = Tensor::ones((2, 2), DType::F64, device)?;
let r = Tensor::randn(0f32, 1.0, (3, 3), device)?;   // normal(mean, std)
let u = Tensor::rand(0f32, 1.0, (2, 5), device)?;     // uniform [lo, hi)
```

## Tensor Info

```rust
t.shape()     // &Shape — dimensions
t.dims()      // &[usize] — dimension sizes as slice
t.dtype()     // DType — F32, F64, U32, etc.
t.device()    // &Device
t.rank()      // number of dimensions
t.elem_count() // total number of elements
```

## Arithmetic Operations

Operations return `Result<Tensor>` — use `?` to unwrap.

```rust
let sum  = (&a + &b)?;    // element-wise add
let diff = (&a - &b)?;    // element-wise subtract
let prod = (&a * &b)?;    // element-wise multiply
let quot = (&a / &b)?;    // element-wise divide

// Scalar operations
let scaled = (&a * 2.0)?;
let shifted = (&a + 1.0)?;

// Matrix multiply
let result = a.matmul(&b)?;   // a @ b

// Unary
let sq = t.sqr()?;
let s  = t.sqrt()?;
let e  = t.exp()?;
let l  = t.log()?;
let r  = t.relu()?;
let n  = t.neg()?;
```

## Reshape and Indexing

```rust
// Reshape
let r = t.reshape((3, 2))?;
let f = t.flatten_all()?;
let u = t.unsqueeze(0)?;      // add dim at position 0
let s = t.squeeze(0)?;        // remove dim at position 0
let p = t.transpose(0, 1)?;   // swap dims

// Indexing with .i()
let row = t.i(0)?;             // first row
let col = t.i((.., 1))?;      // second column
let sub = t.i((1..3, ..))?;   // rows 1-2, all columns
let elm = t.i((0, 2))?;       // element at (0, 2)

// Narrow: extract a slice along a dim
let s = t.narrow(1, 0, 2)?;   // dim 1, start 0, len 2
```

## Reductions

```rust
let m = t.mean_all()?;     // scalar mean of all elements
let s = t.sum_all()?;      // scalar sum of all elements
let s = t.sum(1)?;         // sum along dim 1
let m = t.max(0)?;         // max along dim 0
let m = t.min(0)?;         // min along dim 0
let a = t.argmax(1)?;      // index of max along dim 1
```

## Concatenation

```rust
let cat = Tensor::cat(&[&a, &b], 0)?;   // along dim 0
let cat = Tensor::cat(&[&a, &b], 1)?;   // along dim 1
```

## Converting to Rust Values

```rust
let v: f32   = t.to_scalar()?;        // 0-dim tensor -> scalar
let v: Vec<f32> = t.to_vec1()?;       // 1D tensor -> Vec
let v: Vec<Vec<f32>> = t.to_vec2()?;  // 2D tensor -> nested Vec
```

## Error Handling

Candle operations return `candle_core::Result<T>`. Use `anyhow` for ergonomic error handling:

```rust
fn main() -> anyhow::Result<()> {
    let t = Tensor::new(&[1.0f32], &Device::Cpu)?;
    Ok(())
}
```

## Trainable Variables (Var)

`Var` wraps a tensor and tracks it for gradient computation.

```rust
// Create a trainable variable
let w = Var::from_tensor(&Tensor::zeros((3, 1), DType::F32, device)?)?;

// Use in computation (get underlying tensor)
let pred = x.matmul(w.as_tensor())?;

// After backward pass, manually update
let grads = loss.backward()?;
let w_grad = grads.get(w.as_tensor())?;
w.set(&(w.as_tensor() - (w_grad * lr)?)?)?;
```

## VarMap & VarBuilder

`VarMap` manages all trainable parameters. `VarBuilder` creates parameters within a namespace.

```rust
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);

// vb.pp("name") creates a sub-namespace for parameter naming
let layer = linear(in_features, out_features, vb.pp("layer1"))?;

// Get all variables for optimizer
let all_vars = varmap.all_vars();
```

## Neural Network Layers

```rust
use candle_nn::{linear, Linear};

// linear(in_features, out_features, vb) -> Result<Linear>
let layer = linear(784, 128, vb.pp("fc1"))?;

// Use the layer
let output = layer.forward(&input)?;
```

## Module Trait

Implement `Module` for custom models:

```rust
use candle_core::Module;

struct MyModel {
    layer1: Linear,
    layer2: Linear,
}

impl MyModel {
    fn new(vb: VarBuilder) -> anyhow::Result<Self> {
        let layer1 = linear(2, 16, vb.pp("layer1"))?;
        let layer2 = linear(16, 1, vb.pp("layer2"))?;
        Ok(Self { layer1, layer2 })
    }
}

impl Module for MyModel {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let x = self.layer1.forward(xs)?;
        let x = x.relu()?;
        self.layer2.forward(&x)
    }
}
```

## Activations

```rust
// Method style
let x = x.relu()?;
let x = x.tanh()?;
let x = x.gelu()?;

// Using Activation enum (useful as struct field)
use candle_nn::Activation;
let act = Activation::Gelu;
let x = x.apply(&act)?;  // requires `use candle_core::Module;`
```

## Optimizers

```rust
use candle_nn::{AdamW, SGD, Optimizer, ParamsAdamW};

// SGD
let mut opt = SGD::new(varmap.all_vars(), learning_rate)?;

// AdamW with defaults
let mut opt = AdamW::new(varmap.all_vars(), Default::default())?;

// AdamW with custom params
let params = ParamsAdamW {
    lr: 0.001,
    weight_decay: 0.01,
    ..Default::default()
};
let mut opt = AdamW::new(varmap.all_vars(), params)?;

// Training step (computes backward + updates params)
opt.backward_step(&loss)?;
```

## Loss Functions

```rust
use candle_nn::loss;

// MSE (manual)
let mse = (&pred - &target)?.sqr()?.mean_all()?;

// Cross-entropy (for classification)
// logits: (batch, num_classes), targets: (batch,) of u32 class indices
let ce = loss::cross_entropy(&logits, &targets)?;
```

## Training Loop Template

```rust
fn main() -> anyhow::Result<()> {
    let device = &Device::Cpu;

    // 1. Prepare data
    let x_train = Tensor::new(/* ... */, device)?;
    let y_train = Tensor::new(/* ... */, device)?;

    // 2. Build model
    let varmap = VarMap::new();
    let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
    let model = MyModel::new(vb)?;

    // 3. Set up optimizer
    let mut optimizer = AdamW::new(varmap.all_vars(), Default::default())?;

    // 4. Training loop
    for epoch in 0..num_epochs {
        let pred = model.forward(&x_train)?;
        let loss = /* compute loss */;
        optimizer.backward_step(&loss)?;

        if epoch % 100 == 0 {
            let loss_val: f32 = loss.to_scalar()?;
            println!("epoch {epoch}: loss = {loss_val:.4}");
        }
    }

    Ok(())
}
```
