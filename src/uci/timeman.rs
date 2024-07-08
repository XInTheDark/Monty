pub fn get_time(time: u64, increment: Option<i32>, ply: u16, movestogo: u64) -> u128 {
    let mut max_time;
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
        let opt_constant = (0.0048 + 0.00032 * log_time).min(0.0060);
        let opt_scale = (0.0125 + (ply as f64 + 2.5).sqrt() * opt_constant).min(0.25 * time as f64 / time_left);
        // More time at the start of the game
        let bonus = if ply <= 10 { 1.0 + (11.0 - ply as f64).log10() * 0.5 } else { 1.0 };
        max_time = (opt_scale * bonus * time_left) as u128;
    }
    else {
        max_time = (time / mtg) as u128;
    }

    max_time = max_time.min((time * 850 / 1000) as u128);

    max_time
}