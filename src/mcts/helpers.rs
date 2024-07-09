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
    pub fn get_time(
        time: u64,
        increment: Option<u64>,
        ply: u16,
        movestogo: Option<u64>,
        params: MctsParams,
    ) -> u128 {
        let mut max_time;
        let inc = if let Some(i) = increment { i as u64 } else { 0 };

        // Maximum move horizon of 30 moves
        let tm_mode; // true for increment mode, false for cyclic
        let mtg = if let Some(m) = movestogo {
            tm_mode = false;
            m.min(30).max(1)
        } else {
            tm_mode = true;
            30
        };

        let time_left = (time + inc * (mtg - 1) - 10 * (2 + mtg)).max(1) as f64;

        if tm_mode {
            let log_time = (time_left / 1000.0).log10();
            let opt_constant = (params.tm_opt_value1() / 100.0
                + params.tm_opt_value2() / 1000.0 * log_time)
                .min(params.tm_opt_value3() / 100.0);
            let opt_scale = (params.tm_scale_value1() / 100.0
                + (ply as f64 + params.tm_scale_value2()).powf(params.tm_scale_value3())
                    * opt_constant)
                .min(params.tm_scale_value4() * time as f64 / time_left);
            // More time at the start of the game
            let bonus_ply = params.tm_bonus_ply();
            let bonus = if ply <= bonus_ply as u16 - 1 {
                1.0 + (bonus_ply - ply as f64).log10() * params.tm_bonus_value1()
            } else {
                1.0
            };
            max_time = (opt_scale * bonus * time_left) as u128;
        } else {
            max_time = (time / mtg) as u128;
        }

        max_time = max_time.min((time * params.tm_max_time() as u64 / 1000) as u128);

        max_time
    }
}
