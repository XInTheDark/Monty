use crate::{
    chess::{ChessState, GameState, Move},
    tree::{Node, NodePtr},
};

use super::{SearchHelpers, Searcher};

const TT_SEED_VISITS_CAP: u64 = 64;

pub fn perform_one(
    searcher: &Searcher,
    pos: &mut ChessState,
    ptr: NodePtr,
    depth: &mut usize,
    root_child: &mut Option<NodePtr>,
    thread_id: usize,
) -> Option<(f32, f32)> {
    *depth += 1;

    let cur_hash = pos.hash();
    let tree = searcher.tree;
    let node = &tree[ptr];
    let mut best_move = Move::NULL;

    let mut value = if node.is_terminal() || node.visits() == 0 {
        if node.visits() == 0 {
            node.set_state(pos.game_state());
        }

        // probe hash table to use in place of network
        if node.state() == GameState::Ongoing {
            if let Some(entry) = tree.probe_hash(cur_hash) {
                best_move = entry.best_move();
                tree.seed_node_from_hash(ptr, entry, TT_SEED_VISITS_CAP);
                (entry.q(), entry.d())
            } else {
                get_utility(searcher, ptr, pos)
            }
        } else {
            get_utility(searcher, ptr, pos)
        }
    } else {
        // expand node on the second visit
        if node.is_not_expanded() {
            tree.expand_node(
                ptr,
                pos,
                searcher.params,
                searcher.policy,
                *depth,
                thread_id,
            )?;
        }

        // this node has now been accessed so we need to move its
        // children across if they are in the other tree half
        tree.fetch_children(ptr, thread_id)?;

        // select action to take via PUCT
        let stm = pos.stm();
        let action = pick_action(searcher, ptr, node);

        let child_ptr = node.actions() + action;
        if ptr == searcher.tree.root_node() {
            *root_child = Some(child_ptr);
        }

        let mov = tree[child_ptr].parent_move();
        best_move = mov;

        pos.make_move(mov);
        tree[child_ptr].inc_threads();

        // acquire lock to avoid issues with desynced setting of
        // game state between threads when threads > 1
        let lock = if tree[child_ptr].visits() == 0 {
            Some(node.actions_mut())
        } else {
            None
        };

        // descend further
        let maybe_u = perform_one(searcher, pos, child_ptr, depth, root_child, thread_id);

        drop(lock);

        tree[child_ptr].dec_threads();

        let u = maybe_u?;

        if tree[child_ptr].state() == GameState::Ongoing {
            tree.update_butterfly(stm, mov, u.0, searcher.params);
        }

        tree.propogate_proven_mates(ptr, tree[child_ptr].state());

        u
    };

    let node_visits_before = node.visits();
    let node_draw_before = node.draw();
    let node_parent_q_before = node.q();
    let updated_visits = node_visits_before.saturating_add(1);
    let sample_parent_q = 1.0 - value.0;
    let updated_parent_q = if node_visits_before == 0 {
        sample_parent_q
    } else {
        (node_parent_q_before * node_visits_before as f32 + sample_parent_q) / updated_visits as f32
    };
    let updated_draw =
        (node_draw_before * node_visits_before as f32 + value.1) / updated_visits.max(1) as f32;

    // store an aggregated side-to-move value for the visited node in TT
    tree.push_hash(
        cur_hash,
        1.0 - updated_parent_q,
        updated_draw,
        updated_visits,
        best_move,
        ptr,
    );

    // flip perspective and backpropagate
    value.0 = 1.0 - value.0;
    tree.update_node_stats(ptr, value.0, value.1, thread_id);
    Some(value)
}

fn get_utility(searcher: &Searcher, ptr: NodePtr, pos: &ChessState) -> (f32, f32) {
    match searcher.tree[ptr].state() {
        GameState::Ongoing => {
            let eval = pos.eval_with_contempt(
                searcher.value,
                searcher.params,
                searcher.tree.root_position().stm(),
            );
            (eval.contempt.score(), eval.contempt.draw)
        }
        GameState::Draw => (0.5, 1.0),
        GameState::Lost(_) => (0.0, 0.0),
        GameState::Won(_) => (1.0, 0.0),
    }
}

fn pick_action(searcher: &Searcher, ptr: NodePtr, node: &Node) -> usize {
    let is_root = ptr == searcher.tree.root_node();

    let cpuct = SearchHelpers::get_cpuct(searcher.params, node, is_root);
    let fpu = SearchHelpers::get_fpu(node);
    let expl_scale = SearchHelpers::get_explore_scaling(searcher.params, node);

    let expl = cpuct * expl_scale;

    let actions_ptr = node.actions();
    let mut acc = 0.0;
    let mut k = 0;
    while k < node.num_actions() && acc < searcher.params.policy_top_p() {
        acc += searcher.tree[actions_ptr + k].policy();
        k += 1;
    }
    let mut limit = k.max(searcher.params.min_policy_actions() as usize);
    let mut thresh = 1u64 << (searcher.params.visit_threshold_power() as u32);
    while node.visits() >= thresh && limit < node.num_actions() {
        limit += 2;
        thresh = thresh.checked_shl(1).unwrap_or(u64::MAX);
    }
    limit = limit.min(node.num_actions());

    searcher
        .tree
        .get_best_child_by_key_lim(ptr, limit, |child| {
            let mut q = SearchHelpers::get_action_value(child, fpu);

            // virtual loss
            let threads = f64::from(child.threads());
            if threads > 0.0 {
                let visits = child.visits() as f64;
                let q2 = f64::from(q) * visits
                    / (visits + 1.0 + searcher.params.virtual_loss_weight() * (threads - 1.0));
                q = q2 as f32;
            }

            let u = expl * child.policy() / (1 + child.visits()) as f32;

            q + u
        })
}
