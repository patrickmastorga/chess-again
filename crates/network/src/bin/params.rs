use burn::module::Module;

use chess_again_network::default_device;
use chess_again_network::model::{V1_2M_CONFIG, V1_20M_CONFIG};

fn main() {
    let device = default_device();
    for (name, config) in [
        ("V1_2M_CONFIG", &V1_2M_CONFIG),
        ("V1_20M_CONFIG", &V1_20M_CONFIG),
    ] {
        let model = config.init(&device);
        println!("{name}: {} parameters", model.num_params());
    }
}
