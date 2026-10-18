//! Prints the CUDA devices this node has and checks that Burn runs on them.
//!
//! A smoke test to run before a real training job: it verifies the build
//! actually has a working CUDA backend rather than discovering that an hour
//! into a run. Gated on `cuda`, since there is nothing to check without it.

use anyhow::{Result, bail};
use burn::tensor::{Device, DeviceType, Tensor, TensorData};

fn main() -> Result<()> {
    println!("burn device smoke test");

    let devices = Device::enumerate(DeviceType::Cuda);
    println!("\nenumerated {} CUDA device(s)", devices.len());
    for (index, device) in devices.iter().enumerate() {
        println!("  {index}: {device:?}");
    }
    if devices.is_empty() {
        bail!(
            "no CUDA devices found: is this node's CUDA visible, and was the binary built with --features cuda?"
        );
    }

    println!("\nphysical cards");
    for gpu in Device::enumerate_physical() {
        println!("  {} :: {:?}", gpu.name, gpu.physical);
        println!("      runtimes: {}", gpu.devices.len());
    }

    // Each device gets its own check, so a single bad GPU cannot hide.
    let mut failures = Vec::new();
    for (index, device) in devices.iter().enumerate() {
        match check(device.clone()) {
            Ok(report) => println!("\n[{index}] ok  {report}"),
            Err(error) => {
                println!("\n[{index}] FAILED: {error:#}");
                failures.push(index);
            }
        }
    }

    if !failures.is_empty() {
        bail!(
            "{} of {} device(s) failed: {failures:?}",
            failures.len(),
            devices.len()
        );
    }
    println!("\nall {} device(s) passed", devices.len());
    Ok(())
}

/// Runs a small forward and backward pass on one device, checking the results.
///
/// This is deliberately not just a memset: it covers host-to-device copy,
/// float arithmetic, a reduction across threads, and autodiff, which is where a
/// broken or mismatched backend shows up.
fn check(device: Device) -> Result<String> {
    let device = device.autodiff();

    // [2, 2] @ [2, 2] = [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]]
    //
    // `a` is the leaf we ask for gradients, so it has to be created on the
    // autodiff device and marked with `require_grad`.
    let a = Tensor::<2>::from_data(
        TensorData::new(vec![1.0f32, 2.0, 3.0, 4.0], [2, 2]),
        &device,
    )
    .require_grad();
    let b = Tensor::<2>::from_data(
        TensorData::new(vec![5.0f32, 6.0, 7.0, 8.0], [2, 2]),
        &device,
    );
    let product = a.clone().matmul(b);
    let expected = [19.0f32, 22.0, 43.0, 50.0];
    let got: Vec<f32> = product.clone().into_data().to_vec().map_err(to_anyhow)?;
    if got != expected {
        bail!("matmul gave {got:?}, expected {expected:?}");
    }

    // A reduction that spans the block, so a broken reduce is caught.
    let sum: f32 = product
        .clone()
        .sum()
        .into_data()
        .to_vec::<f32>()
        .map_err(to_anyhow)?[0];
    if (sum - 134.0).abs() > 1e-4 {
        bail!("sum gave {sum}, expected 134");
    }

    // Autodiff: d/dA of sum(A @ B) is B's row sums, i.e. [[11, 15], [11, 15]].
    let grads = product.sum().backward();
    let grad_a: Vec<f32> = a
        .grad(&grads)
        .expect("a is a leaf")
        .into_data()
        .to_vec()
        .map_err(to_anyhow)?;
    let expected_grad = [11.0f32, 15.0, 11.0, 15.0];
    if grad_a != expected_grad {
        bail!("gradient gave {grad_a:?}, expected {expected_grad:?}");
    }

    Ok(format!(
        "{} | matmul ok, sum ok, backward ok",
        describe(&device)
    ))
}

fn describe(device: &Device) -> String {
    let settings = device.settings();
    format!(
        "{:?} (autodiff: {})",
        settings.float_dtype,
        device.is_autodiff()
    )
}

/// `to_vec` reports a `DataError`; the tensor reads report an `ExecutionError`.
/// Both just need to become an `anyhow` error carrying their message.
fn to_anyhow(error: impl std::fmt::Display) -> anyhow::Error {
    anyhow::anyhow!("{error}")
}
