#[cfg(not(test))]
use recon_mcts_test_nim::visualize_2048::run_visualization;

#[cfg(test)]
fn run_visualization() {
    // Stub function for test mode
    unimplemented!("This binary should not be compiled during tests")
}

fn main() {
    println!("🎮 2048 Game with MCTS AI Visualization");
    println!("========================================");

    run_visualization();
}
