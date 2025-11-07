//! Linker Infrastructure
//!
//! This module provides cross-platform linking support for:
//! - Static libraries (.a, .lib)
//! - Shared libraries (.so, .dll, .dylib)
//! - Executables

use crate::error::{TensorError, TensorResult};
use std::path::{Path, PathBuf};
use std::process::Command;

/// Target platform for linking
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Platform {
    Linux,
    MacOS,
    Windows,
}

impl Platform {
    /// Get the current platform
    pub fn current() -> Self {
        if cfg!(target_os = "linux") {
            Platform::Linux
        } else if cfg!(target_os = "macos") {
            Platform::MacOS
        } else if cfg!(target_os = "windows") {
            Platform::Windows
        } else {
            // Default to Linux for unknown platforms
            Platform::Linux
        }
    }

    /// Get the object file extension for this platform
    pub fn object_extension(&self) -> &'static str {
        match self {
            Platform::Windows => "obj",
            _ => "o",
        }
    }

    /// Get the static library extension for this platform
    pub fn static_lib_extension(&self) -> &'static str {
        match self {
            Platform::Windows => "lib",
            _ => "a",
        }
    }

    /// Get the shared library extension for this platform
    pub fn shared_lib_extension(&self) -> &'static str {
        match self {
            Platform::Linux => "so",
            Platform::MacOS => "dylib",
            Platform::Windows => "dll",
        }
    }

    /// Get the executable extension for this platform
    pub fn executable_extension(&self) -> &'static str {
        match self {
            Platform::Windows => "exe",
            _ => "",
        }
    }

    /// Get the static library prefix for this platform
    pub fn static_lib_prefix(&self) -> &'static str {
        match self {
            Platform::Windows => "",
            _ => "lib",
        }
    }

    /// Get the shared library prefix for this platform
    pub fn shared_lib_prefix(&self) -> &'static str {
        match self {
            Platform::Windows => "",
            _ => "lib",
        }
    }
}

/// Linker for creating libraries and executables
pub struct Linker {
    platform: Platform,
}

impl Linker {
    /// Create a new linker for the current platform
    pub fn new() -> Self {
        Linker {
            platform: Platform::current(),
        }
    }

    /// Create a new linker for a specific platform
    pub fn for_platform(platform: Platform) -> Self {
        Linker { platform }
    }

    /// Create a static library from object files
    pub fn create_static_library(
        &self,
        object_files: &[impl AsRef<Path>],
        output_path: impl AsRef<Path>,
    ) -> TensorResult<()> {
        let output_path = output_path.as_ref();

        match self.platform {
            Platform::Linux | Platform::MacOS => {
                // Use ar to create static library
                let mut cmd = Command::new("ar");
                cmd.arg("rcs").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute ar: {}. Make sure binutils is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "ar failed: {}",
                        stderr
                    )));
                }

                println!("Static library created: {}", output_path.display());
                Ok(())
            }
            Platform::Windows => {
                // Use lib.exe (MSVC) or ar (MinGW)
                // Try lib.exe first (MSVC)
                let mut cmd = Command::new("lib");
                cmd.arg(format!("/OUT:{}", output_path.display()));

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output();

                if let Ok(output) = output {
                    if output.status.success() {
                        println!("Static library created: {}", output_path.display());
                        return Ok(());
                    }
                }

                // Fallback to ar (MinGW)
                let mut cmd = Command::new("ar");
                cmd.arg("rcs").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute lib.exe or ar: {}. Make sure MSVC or MinGW is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "Library creation failed: {}",
                        stderr
                    )));
                }

                println!("Static library created: {}", output_path.display());
                Ok(())
            }
        }
    }

    /// Create a shared library from object files
    pub fn create_shared_library(
        &self,
        object_files: &[impl AsRef<Path>],
        output_path: impl AsRef<Path>,
    ) -> TensorResult<()> {
        let output_path = output_path.as_ref();

        match self.platform {
            Platform::Linux => {
                // Use gcc or ld to create shared library
                let mut cmd = Command::new("gcc");
                cmd.arg("-shared").arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute gcc: {}. Make sure gcc is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "gcc failed: {}",
                        stderr
                    )));
                }

                println!("Shared library created: {}", output_path.display());
                Ok(())
            }
            Platform::MacOS => {
                // Use clang to create dylib
                let mut cmd = Command::new("clang");
                cmd.arg("-dynamiclib").arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute clang: {}. Make sure Xcode Command Line Tools are installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "clang failed: {}",
                        stderr
                    )));
                }

                println!("Shared library created: {}", output_path.display());
                Ok(())
            }
            Platform::Windows => {
                // Use link.exe (MSVC) or gcc (MinGW)
                // Try link.exe first (MSVC)
                let mut cmd = Command::new("link");
                cmd.arg("/DLL")
                    .arg(format!("/OUT:{}", output_path.display()));

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output();

                if let Ok(output) = output {
                    if output.status.success() {
                        println!("Shared library created: {}", output_path.display());
                        return Ok(());
                    }
                }

                // Fallback to gcc (MinGW)
                let mut cmd = Command::new("gcc");
                cmd.arg("-shared").arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute link.exe or gcc: {}. Make sure MSVC or MinGW is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "Shared library creation failed: {}",
                        stderr
                    )));
                }

                println!("Shared library created: {}", output_path.display());
                Ok(())
            }
        }
    }

    /// Create an executable from object files
    pub fn create_executable(
        &self,
        object_files: &[impl AsRef<Path>],
        output_path: impl AsRef<Path>,
    ) -> TensorResult<()> {
        let output_path = output_path.as_ref();

        match self.platform {
            Platform::Linux => {
                // Use gcc to create executable
                let mut cmd = Command::new("gcc");
                cmd.arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute gcc: {}. Make sure gcc is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "gcc failed: {}",
                        stderr
                    )));
                }

                println!("Executable created: {}", output_path.display());
                Ok(())
            }
            Platform::MacOS => {
                // Use clang to create executable
                let mut cmd = Command::new("clang");
                cmd.arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute clang: {}. Make sure Xcode Command Line Tools are installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "clang failed: {}",
                        stderr
                    )));
                }

                println!("Executable created: {}", output_path.display());
                Ok(())
            }
            Platform::Windows => {
                // Use link.exe (MSVC) or gcc (MinGW)
                // Try link.exe first (MSVC)
                let mut cmd = Command::new("link");
                cmd.arg(format!("/OUT:{}", output_path.display()));

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output();

                if let Ok(output) = output {
                    if output.status.success() {
                        println!("Executable created: {}", output_path.display());
                        return Ok(());
                    }
                }

                // Fallback to gcc (MinGW)
                let mut cmd = Command::new("gcc");
                cmd.arg("-o").arg(output_path);

                for obj in object_files {
                    cmd.arg(obj.as_ref());
                }

                let output = cmd.output().map_err(|e| {
                    TensorError::CompilationError(format!(
                        "Failed to execute link.exe or gcc: {}. Make sure MSVC or MinGW is installed.",
                        e
                    ))
                })?;

                if !output.status.success() {
                    let stderr = String::from_utf8_lossy(&output.stderr);
                    return Err(TensorError::CompilationError(format!(
                        "Executable creation failed: {}",
                        stderr
                    )));
                }

                println!("Executable created: {}", output_path.display());
                Ok(())
            }
        }
    }

    /// Get the platform this linker targets
    pub fn platform(&self) -> Platform {
        self.platform
    }
}

impl Default for Linker {
    fn default() -> Self {
        Self::new()
    }
}
