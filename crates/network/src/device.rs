//! Device-selection module for the whole network stack.
//!
//! Burn 0.22 removed the backend type parameter from models and tensors.
//! A [`Device`] now identifies where and how operations execute, so the
//! backend is chosen at runtime instead of through the type system. Model
//! and dataset code is written once; only the constructor below names a
//! concrete backend.
//!
//! Each `Device` constructor requires the matching feature on `burn`; see
//! the Burn Book, "Backend and Device", for the full constructor table.

use burn::tensor::Device;

/// The device used by the model, datasets and tests.
pub fn default_device() -> Device {
    #[cfg(feature = "cuda")]
    {
        Device::cuda(0)
    }

    #[cfg(all(feature = "metal", not(feature = "cuda")))]
    {
        Device::metal(Default::default())
    }

    #[cfg(all(feature = "webgpu", not(any(feature = "cuda", feature = "metal"))))]
    {
        Device::webgpu(Default::default())
    }

    #[cfg(not(any(feature = "cuda", feature = "metal", feature = "webgpu")))]
    {
        Device::flex()
    }
}
