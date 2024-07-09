use crate::{mcts::MctsParams, tree::Edge};

pub struct SearchHelpers;

impl SearchHelpers {
    /// CPUCT
    ///
    /// Larger value implies more exploration.
    pub fn get_cpuct(params: &MctsParams, parent: &Edge, is_root: bool) -> f32 {
        // baseline CPUCT value
        let mut cpuct = if is_root {
            params.root_cpuct()
        } else {
            params.cpuct()
        };

        // scale CPUCT as visits increase
        let scale = params.cpuct_visits_scale() * 128.0;
        cpuct *= 1.0 + ((parent.visits() as f32 + scale) / scale).ln();

        // scale CPUCT with variance of Q
        if parent.visits() > 1 {
            let frac = parent.var().sqrt() / params.cpuct_var_scale();
            cpuct *= 1.0 + params.cpuct_var_weight() * (frac - 1.0);
        }

        cpuct
    }

    /// Exploration Scaling
    ///
    /// Larger value implies more exploration.
    pub fn get_explore_scaling(params: &MctsParams, parent: &Edge) -> f32 {
        (params.expl_tau() * (parent.visits().max(1) as f32).ln()).exp()
    }

    /// First Play Urgency
    ///
    /// #### Note
    /// Must return a value in [0, 1].
    pub fn get_fpu(parent: &Edge) -> f32 {
        1.0 - parent.q()
    }

    /// Get a predicted win probability for an action
    ///
    /// #### Note
    /// Must return a value in [0, 1].
    pub fn get_action_value(action: &Edge, fpu: f32) -> f32 {
        if action.visits() == 0 {
            fpu
        } else {
            action.q()
        }
    }

    /// Calculates the maximum allowed time usage for a search
    ///
    /// #### Note
    /// This will be overriden by a `go movetime` command,
    /// and a move overhead will be applied to this, so no
    /// need for it here.
    pub fn get_time(time: u64, increment: Option<u64>, ply: u16, movestogo: Option<u64>) -> u128 {
        let mut max_time;
        let inc =
            if let Some(i) = increment {
                i as u64
            } else {
                0
            };

        // Maximum move horizon of 30 moves
        let tm_mode; // true for increment mode, false for cyclic
        let mut mtg = if let Some(m) = movestogo {
            tm_mode = false;
            m.min(30).max(1)
        } else {
            tm_mode = true;
            30
        };

        // If less than one second, gradually reduce mtg
        if time < 1000 && mtg as f32 / inc as f32 > 0.03 {
            mtg = (time as f64 * 0.03).max(2.0) as u64;
        }

        let time_left = (time + inc * (mtg - 1) - 10 * (2 + mtg)).max(1) as f64;

        if tm_mode {
            let log_time = (time_left / 1000.0).log10();
            let opt_constant = (0.0048 + 0.00032 * log_time).min(0.0060);
            let opt_scale = (0.0125 + (ply as f64 + 2.5).sqrt() * opt_constant).min(0.25 * time as f64 / time_left);
            // More time at the start of the game
            let bonus = if ply <= 10 { 1.0 + (11.0 - ply as f64).log10() * 0.5 } else { 1.0 };
            max_time = (opt_scale * bonus * time_left) as u128;
        } else {
            max_time = (time / mtg) as u128;
        }

        max_time = max_time.min((time * 850 / 1000) as u128);

        max_time
    }
}
