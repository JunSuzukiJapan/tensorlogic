//! Metal勾配計算用の共通ヘルパー関数

use crate::device::{Device, MetalBuffer};
use crate::error::{TensorError, TensorResult};
use crate::tensor::Tensor;

/// シンプルな要素ごとのMetal勾配カーネルを実行
///
/// # 引数
/// * `kernel_name` - 実行するカーネル関数名
/// * `grad_output` - 出力の勾配
/// * `input_buffers` - 入力バッファのスライス（最大3つまで）
///
/// # 戻り値
/// 計算された入力の勾配テンソル
pub fn execute_simple_metal_gradient(
    kernel_name: &str,
    grad_output: &Tensor,
    input_buffers: &[&MetalBuffer],
) -> TensorResult<Tensor> {
    let grad_output_buf = grad_output.buffer().as_metal()?;

    let mut device = match grad_output.device() {
        Device::Metal(dev) => dev.clone(),
        _ => {
            return Err(TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            ))
        }
    };

    // シェーダーライブラリをロード
    if device.library().is_none() {
        let shader_source = include_str!("../../../shaders/gradients.metal");
        device.load_library(shader_source)?;
    }

    // 結果バッファを作成
    let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), grad_output.numel())?;

    // パイプライン取得
    let library_ref = device.library();
    let library = library_ref
        .as_ref()
        .ok_or_else(|| TensorError::MetalError("Library not loaded".to_string()))?;
    let pipeline = library
        .get_function(kernel_name, None)
        .map_err(|e| {
            TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
        })?;

    let pipeline_state = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&pipeline)
        .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {:?}", e)))?;

    // コマンド実行
    let command_queue = device.command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);

    // バッファ設定
    encoder.set_buffer(0, Some(grad_output_buf.metal_buffer()), 0);
    for (i, buf) in input_buffers.iter().enumerate() {
        encoder.set_buffer((i + 1) as u64, Some(buf.metal_buffer()), 0);
    }
    encoder.set_buffer((input_buffers.len() + 1) as u64, Some(result_buf.metal_buffer()), 0);

    // グリッドサイズとスレッドグループサイズ
    let grid_size = metal::MTLSize::new(grad_output.numel() as u64, 1, 1);
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Tensor::new(
        crate::tensor::BufferHandle::Metal(result_buf),
        grad_output.shape().clone(),
        grad_output.device().clone(),
    )
}

/// パラメータ付きのMetal勾配カーネルを実行
///
/// # 引数
/// * `kernel_name` - 実行するカーネル関数名
/// * `grad_output` - 出力の勾配
/// * `input_buffers` - 入力バッファのスライス
/// * `param_buffers` - パラメータバッファのスライス（スカラー値など）
///
/// # 戻り値
/// 計算された入力の勾配テンソル
pub fn execute_parametric_metal_gradient(
    kernel_name: &str,
    grad_output: &Tensor,
    input_buffers: &[&MetalBuffer],
    param_buffers: &[&MetalBuffer],
) -> TensorResult<Tensor> {
    let grad_output_buf = grad_output.buffer().as_metal()?;

    let mut device = match grad_output.device() {
        Device::Metal(dev) => dev.clone(),
        _ => {
            return Err(TensorError::DeviceConversionError(
                "Not on Metal device".to_string(),
            ))
        }
    };

    // シェーダーライブラリをロード
    if device.library().is_none() {
        let shader_source = include_str!("../../../shaders/gradients.metal");
        device.load_library(shader_source)?;
    }

    // 結果バッファを作成
    let result_buf = MetalBuffer::new_uninit_pooled(device.buffer_pool(), grad_output.numel())?;

    // パイプライン取得
    let library_ref = device.library();
    let library = library_ref
        .as_ref()
        .ok_or_else(|| TensorError::MetalError("Library not loaded".to_string()))?;
    let pipeline = library
        .get_function(kernel_name, None)
        .map_err(|e| {
            TensorError::MetalError(format!("Failed to get kernel {}: {:?}", kernel_name, e))
        })?;

    let pipeline_state = device
        .metal_device()
        .new_compute_pipeline_state_with_function(&pipeline)
        .map_err(|e| TensorError::MetalError(format!("Failed to create pipeline: {:?}", e)))?;

    // コマンド実行
    let command_queue = device.command_queue();
    let command_buffer = command_queue.new_command_buffer();
    let encoder = command_buffer.new_compute_command_encoder();

    encoder.set_compute_pipeline_state(&pipeline_state);

    // バッファ設定: grad_output, inputs..., params..., result
    let mut buffer_idx = 0u64;

    encoder.set_buffer(buffer_idx, Some(grad_output_buf.metal_buffer()), 0);
    buffer_idx += 1;

    for buf in input_buffers {
        encoder.set_buffer(buffer_idx, Some(buf.metal_buffer()), 0);
        buffer_idx += 1;
    }

    for buf in param_buffers {
        encoder.set_buffer(buffer_idx, Some(buf.metal_buffer()), 0);
        buffer_idx += 1;
    }

    encoder.set_buffer(buffer_idx, Some(result_buf.metal_buffer()), 0);

    // グリッドサイズとスレッドグループサイズ
    let grid_size = metal::MTLSize::new(grad_output.numel() as u64, 1, 1);
    let threadgroup_size = metal::MTLSize::new(256, 1, 1);

    encoder.dispatch_threads(grid_size, threadgroup_size);
    encoder.end_encoding();
    command_buffer.commit();
    command_buffer.wait_until_completed();

    Tensor::new(
        crate::tensor::BufferHandle::Metal(result_buf),
        grad_output.shape().clone(),
        grad_output.device().clone(),
    )
}
