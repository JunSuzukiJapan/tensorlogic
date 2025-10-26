//! Tensor I/O methods (device transfer, save/load, data access)

use crate::device::{Device, MetalBuffer, MetalDevice};
use crate::error::{TensorError, TensorResult};
use crate::tensor::{BufferHandle, FloatType, Tensor};
use crate::tensor::{TensorAccessors, TensorCreation, TensorIO, TensorTransform};

/// Trait for tensor I/O operations
pub trait TensorIO<T: FloatType>: Sized {
    /// Transfer tensor to CPU
    fn to_cpu(&self) -> TensorResult<Self>;

    /// Transfer tensor to Metal device
    fn to_metal(&self, device: &MetalDevice) -> TensorResult<Self>;

    /// Get data as Vec<T> (copies from GPU if needed)
    fn to_vec(&self) -> Vec<T>;

    /// Get data as Vec<f32> (copies from GPU if needed)
    fn to_vec_f32(&self) -> Vec<f32>;

    /// Save tensor to a binary file
    ///
    /// Format: [num_dims: u32][dim0: u32][dim1: u32]...[type_size: u32][data: T...]
    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> TensorResult<()>;

    /// Load tensor from a binary file
    ///
    /// Format: [num_dims: u32][dim0: u32][dim1: u32]...[type_size: u32][data: T...]
    fn load<P: AsRef<std::path::Path>>(device: &Device, path: P) -> TensorResult<Self>;
}

impl<T: FloatType> TensorIO<T> for Tensor<T> {
    fn to_cpu(&self) -> TensorResult<Self> {
        if self.buffer.is_cpu() {
            return Ok(self.clone());
        }

        let data = self.buffer.to_cpu_vec();
        let buffer = BufferHandle::CPU(data);

        Self::new(buffer, self.shape.clone(), Device::CPU)
    }

    fn to_metal(&self, device: &MetalDevice) -> TensorResult<Self> {
        if let BufferHandle::Metal(_) = &self.buffer {
            if let Device::Metal(dev) = &self.device {
                if dev == device {
                    return Ok(self.clone());
                }
            }
        }

        let data = self.buffer.to_cpu_vec();
        let metal_buffer = MetalBuffer::from_slice(device.metal_device(), &data)?;
        let buffer = BufferHandle::Metal(metal_buffer);

        Self::new(buffer, self.shape.clone(), Device::Metal(device.clone()))
    }

    fn to_vec(&self) -> Vec<T> {
        self.buffer.to_cpu_vec()
    }

    fn to_vec_f32(&self) -> Vec<f32> {
        self.buffer.to_cpu_vec().iter().map(|x| x.to_f32()).collect()
    }

    fn save<P: AsRef<std::path::Path>>(&self, path: P) -> TensorResult<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(path).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to create file: {}", e)
        ))?;

        // Write number of dimensions
        let num_dims = self.dims().len() as u32;
        file.write_all(&num_dims.to_le_bytes()).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to write dimensions count: {}", e)
        ))?;

        // Write each dimension
        for &dim in self.dims() {
            file.write_all(&(dim as u32).to_le_bytes()).map_err(|e| TensorError::InvalidOperation(
                format!("Failed to write dimension: {}", e)
            ))?;
        }

        // Write type size for validation
        let type_size = T::size_in_bytes() as u32;
        file.write_all(&type_size.to_le_bytes()).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to write type size: {}", e)
        ))?;

        // Write tensor data
        let data = self.to_vec();
        let bytes: Vec<u8> = unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const u8,
                data.len() * T::size_in_bytes()
            ).to_vec()
        };
        file.write_all(&bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to write tensor data: {}", e)
        ))?;

        Ok(())
    }

    fn load<P: AsRef<std::path::Path>>(device: &Device, path: P) -> TensorResult<Self> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to open file: {}", e)
        ))?;

        // Read number of dimensions
        let mut num_dims_bytes = [0u8; 4];
        file.read_exact(&mut num_dims_bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to read dimensions count: {}", e)
        ))?;
        let num_dims = u32::from_le_bytes(num_dims_bytes) as usize;

        // Read each dimension
        let mut shape = Vec::with_capacity(num_dims);
        for _ in 0..num_dims {
            let mut dim_bytes = [0u8; 4];
            file.read_exact(&mut dim_bytes).map_err(|e| TensorError::InvalidOperation(
                format!("Failed to read dimension: {}", e)
            ))?;
            shape.push(u32::from_le_bytes(dim_bytes) as usize);
        }

        // Read and validate type size
        let mut type_size_bytes = [0u8; 4];
        file.read_exact(&mut type_size_bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to read type size: {}", e)
        ))?;
        let file_type_size = u32::from_le_bytes(type_size_bytes) as usize;

        if file_type_size != T::size_in_bytes() {
            return Err(TensorError::InvalidOperation(
                format!("Type size mismatch: file has {}, expected {}",
                    file_type_size, T::size_in_bytes())
            ));
        }

        // Calculate expected data size
        let numel: usize = shape.iter().product();
        let expected_bytes = numel * T::size_in_bytes();

        // Read tensor data
        let mut bytes = vec![0u8; expected_bytes];
        file.read_exact(&mut bytes).map_err(|e| TensorError::InvalidOperation(
            format!("Failed to read tensor data: {}", e)
        ))?;

        // Convert bytes to T vector
        let data: Vec<T> = unsafe {
            std::slice::from_raw_parts(
                bytes.as_ptr() as *const T,
                numel
            ).to_vec()
        };

        // Create tensor on specified device
        match device {
            Device::Metal(metal_device) => Self::from_vec_metal(metal_device, data, shape),
            Device::CPU => Self::from_vec(data, shape),
            Device::NeuralEngine => {
                // NeuralEngine uses Metal backend for tensor storage
                // We need a Metal device - for now, create a new one
                let metal = MetalDevice::new()?;
                Self::from_vec_metal(&metal, data, shape)
            }
        }
    }
}
