//! Profiles DDP training across 1, 2, 4 and 8 devices.
//!
//! For each device count it times the dataloader alone against the real
//! training loop, which says whether the run is limited by data or by the
//! devices. DDP needs a collective-capable backend, so this binary declares
//! `required-features = ["cuda"]`.
//!
//! ```text
//! cargo run --release --features cuda -p chess-again-train --bin v1_profile_ddp
//! ```

use anyhow::Result;
use burn::tensor::{Device, DeviceType};
use chess_again_train::v1_profile_ddp;

const DEVICE_SWEEP: &[usize] = &[1, 2, 4, 8];
const BATCH_SWEEP: &[usize] = &[512, 1024, 2048, 4096];
const STEPS: u64 = 100;

/// Dataloader workers.
const DATA_WORKERS: usize = 4;
const BATCH_WORKERS: usize = 2;

fn main() -> Result<()> {
    let data_dir = "data/train";

    let available = Device::enumerate(DeviceType::Cuda).autodiff().into_vec();
    eprintln!("{} CUDA device(s) available", available.len());

    let max = DEVICE_SWEEP.iter().copied().max().unwrap_or(1);
    if available.len() < max {
        eprintln!(
            "warning: {} device(s) available but the sweep wants up to {max}; \
             skipping the larger counts",
            available.len()
        );
    }

    for &count in DEVICE_SWEEP {
        if count > available.len() {
            eprintln!(
                "\nskipping {count} devices: only {} available",
                available.len()
            );
            continue;
        }
        eprintln!("\n=== {count} device(s) ===");
        let devices = available[..count].to_vec();
        for &batch_size in BATCH_SWEEP {
            eprintln!("\n--- Batch size: {batch_size} ---");
            v1_profile_ddp(
                &devices,
                DATA_WORKERS,
                BATCH_WORKERS,
                batch_size,
                STEPS,
                data_dir,
            )?;
        }
    }
    Ok(())
}
