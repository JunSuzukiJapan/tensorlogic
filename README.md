# TensorLogic

An experimental implementation of [TensorLogic](https://tensor-logic.org/) - a unified tensor algebra and logic programming language.

## About

This project aims to implement TensorLogic, which combines:
- **Tensor operations** (neural networks, deep learning)
- **Logic programming** (symbolic AI, reasoning)
- **f16-only operations** on Apple Silicon (Metal GPU + Neural Engine)

## Status

🚧 **Experimental** - Early development phase

Currently implemented:
- ✅ f16 tensor operations on Metal GPU
- ✅ Basic arithmetic operations (add, sub, mul, div)
- ✅ Device management (Metal/CPU)
- ⏳ Neural Engine integration (planned)
- ⏳ Logic programming (planned)

## Quick Start

```rust
use tensorlogic::{MetalDevice, Tensor};

// Initialize Metal device
let device = MetalDevice::new()?;

// Create tensors
let a = Tensor::ones(&device, vec![3])?;
let b = Tensor::from_vec_metal(&device, vec![2.0, 3.0, 4.0], vec![3])?;

// Perform operations
let c = a.add(&b)?; // [3.0, 4.0, 5.0]
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
- MIT license ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## References

- [TensorLogic Paper](https://tensor-logic.org/)
- Pedro Domingos - "TensorLogic: The Language of AI"
