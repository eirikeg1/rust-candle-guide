# Task 3: Implement IrisNet::new()

## Hint

Same pattern as Exercise 4 and 5: use `candle_nn::linear()` with `vb.pp("name")` for each layer. The architecture is 4 → 16 → 16 → 3: four input features (sepal/petal length/width), two hidden layers of 16 units, and three output classes (setosa, versicolor, virginica).

## Solution

```rust
let layer1 = linear(4, 16, vb.pp("layer1"))?;
let layer2 = linear(16, 16, vb.pp("layer2"))?;
let layer3 = linear(16, 3, vb.pp("layer3"))?;
Ok(Self { layer1, layer2, layer3 })
```
