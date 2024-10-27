fn main() {
    let mut args = std::env::args();
    args.next();
    let buffer_size_mb = args.next().unwrap().parse().unwrap();
    let threads = args.next().unwrap().parse().unwrap();

    policy::train(
        buffer_size_mb,
        threads,
        "/home/neural/policy_data/interleaved.binpack".to_string(),
        540,
        0.001,
        0.000001,
        540,
    );
}
