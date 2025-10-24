//! Math operations for TensorLogic interpreter

use super::*;
use crate::tensor::Tensor;

impl Interpreter {
    pub(super) fn eval_math_function(&mut self, name: &str, args: &[TensorExpr]) -> Option<RuntimeResult<Value>> {
        match name {
            "matmul" => Some(self.eval_matmul(args)),
            "sigmoid" => Some(self.eval_sigmoid(args)),
            "sum" | "mean" | "max" | "min" | "pow" |
            "relu" | "gelu" | "tanh" | "exp" | "log" | "sqrt" |
            "sin" | "cos" | "tan" => {
                Some(Err(RuntimeError::NotImplemented(
                    format!("Math function '{}' migration in progress", name)
                )))
            }
            _ => None,
        }
    }

    /// matmul(a, b) -> tensor
    /// Matrix multiplication: a @ b
    fn eval_matmul(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 2 {
            return Err(RuntimeError::TypeError(
                format!("matmul() expects 2 arguments (a, b), got {}", args.len())
            ));
        }

        // Evaluate both tensor arguments
        let a_val = self.eval_expr(&args[0])?;
        let a = a_val.as_tensor()?;

        let b_val = self.eval_expr(&args[1])?;
        let b = b_val.as_tensor()?;

        // Perform matrix multiplication
        let result = a.matmul(&b)
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }

    /// sigmoid(x) -> tensor
    /// Sigmoid activation: σ(x) = 1 / (1 + exp(-x))
    fn eval_sigmoid(&mut self, args: &[TensorExpr]) -> RuntimeResult<Value> {
        if args.len() != 1 {
            return Err(RuntimeError::TypeError(
                format!("sigmoid() expects 1 argument (tensor), got {}", args.len())
            ));
        }

        // Evaluate tensor argument
        let tensor_val = self.eval_expr(&args[0])?;
        let tensor = tensor_val.as_tensor()?;

        // Apply sigmoid
        let result = tensor.sigmoid()
            .map_err(|e| RuntimeError::TensorError(e))?;

        Ok(Value::Tensor(result))
    }
}
