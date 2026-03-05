# Task 4: Model and optimizer setup

## Hint

Before the training loop, you need to set up three things:

1. **VarMap** — a container that holds all trainable parameters. Created with `VarMap::new()`.
2. **VarBuilder** — a factory for creating parameters inside the VarMap. Created with `VarBuilder::from_varmap(&varmap, DType::F32, device)`.
3. **Model** — your `Mlp` struct, initialized via `Mlp::new(vb)?`. This registers its layers' parameters in the VarMap.
4. **Optimizer** — `AdamW` takes all variables from the VarMap and manages their updates. Created with `candle_nn::AdamW::new(varmap.all_vars(), Default::default())?`.

This setup is already written for you in the exercise code. Make sure you understand how the pieces connect: the VarMap owns the parameters, the model uses them for computation, and the optimizer updates them.

## Key Pattern

```rust
let varmap = VarMap::new();
let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
let model = Mlp::new(vb)?;
let mut optimizer = candle_nn::AdamW::new(varmap.all_vars(), Default::default())?;
```
