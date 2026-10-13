//! Single backend-selection module for the whole network stack.
//!
//! All model/dataset code must stay backend-generic and reference
//! [`DefaultBackend`] from here instead of naming a backend directly.
//!
//! Selection is driven by cargo features:
//! - default & tests: `burn-flex` (CPU, eager, deterministic)
//! - training (M5):   feature `metal` (CubeCL Metal compiler)
//! - later cluster:   feature `cuda` (A100)
//! - later wasm ship: `webgpu` + flex (inference only)

/// The backend used by the V1_0 model, datasets and tests.
#[cfg(feature = "cuda")]
pub type DefaultBackend = burn::backend::Cuda;

#[cfg(all(feature = "metal", not(feature = "cuda")))]
pub type DefaultBackend = burn::backend::wgpu::Metal;

#[cfg(all(feature = "webgpu", not(any(feature = "cuda", feature = "metal"))))]
pub type DefaultBackend = burn::backend::wgpu::WebGpu;

#[cfg(not(any(feature = "metal", feature = "cuda")))]
pub type DefaultBackend = burn::backend::Flex;

#[cfg(feature = "cuda")]
pub fn default_device() -> burn::backend::cuda::CudaDevice {
    burn::backend::cuda::CudaDevice::new(0)
}

#[cfg(all(any(feature = "metal", feature = "webgpu"), not(feature = "cuda")))]
pub fn default_device() -> burn::backend::wgpu::WgpuDevice {
    burn::backend::wgpu::WgpuDevice::default() // Metal adapter on macOS
}

#[cfg(not(any(feature = "cuda", feature = "metal", feature = "webgpu")))]
pub fn default_device() -> burn::backend::flex::FlexDevice {
    burn::backend::flex::FlexDevice::default()
}
