pub fn get_time(time: u64, increment: Option<i32>, movestogo: u64) -> u128 {
    let max_time;
    let inc;
    if let Some(i) = increment {
        inc = i as u64;
    } else {
        inc = 0;
    }

    // Maximum move horizon of 30 moves
    let mut mtg = movestogo.min(30).max(1);

    // If less than one second, gradually reduce mtg
    if time < 1000 && mtg as f32 / inc as f32 > 0.03 {
        mtg = (time as f64 * 0.03).max(2.0) as u64;
    }

    let time_left = (time + inc * (mtg - 1) - 10 * (2 + mtg)).max(1) as f64;

    if movestogo == 0 {
        let log_time = (time_left / 1000.0).log10();
        let opt_constant = (0.00308 + 0.000319 * log_time).min(0.00506);
        let opt_scale = (0.0122 + 3.60 * opt_constant).min(0.213 * time as f64 / time_left);
        max_time = (opt_scale * time_left) as u128;
    }
    else {
        max_time = (time / mtg) as u128;
    }

    max_time
}