use botbowl_ui::render_seeded_snapshot;

fn render(seed: u64, step: usize) -> String {
    render_seeded_snapshot(seed, step, 120, 40).expect("render")
}

#[test]
fn seed_42_step_0() {
    insta::assert_snapshot!(render(42, 0));
}

#[test]
fn seed_42_step_50() {
    insta::assert_snapshot!(render(42, 50));
}

#[test]
fn seed_42_step_500() {
    insta::assert_snapshot!(render(42, 500));
}

#[test]
fn determinism_is_byte_stable() {
    let a = render(7, 123);
    let b = render(7, 123);
    assert_eq!(a, b);
}

#[test]
fn different_seeds_diverge() {
    let a = render(1, 200);
    let b = render(2, 200);
    assert_ne!(a, b);
}
