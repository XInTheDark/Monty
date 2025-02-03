use once_cell::sync::OnceCell;
use std::cell::Cell;
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::chess::ChessState;
use crate::chess::Move;
use crate::mcts::MctsParams;
use crate::networks::{Accumulator, PolicyNetwork, ValueNetwork, POLICY_L1};

/// A request for a value-network evaluation.
pub struct ValueEvalRequest {
    pub board: ChessState,
    pub params: MctsParams,
    pub response: mpsc::Sender<f32>,
}
/// A request for a policy-network evaluation.
pub struct PolicyEvalRequest {
    pub board: ChessState,
    pub mov: Move,
    pub feats: Accumulator<i16, { POLICY_L1 / 2 }>,
    pub response: mpsc::Sender<f32>,
}
/// A BatchEvaluator holds a sender for value and policy evaluation requests.
/// Evaluator threads (spawned by new()) will pop requests, accumulate a batch (up to 16)
/// and then process them (calling the network’s evaluation functions).
pub struct BatchEvaluator {
    pub value_sender: mpsc::Sender<ValueEvalRequest>,
    pub policy_sender: mpsc::Sender<PolicyEvalRequest>,
}
/// Use a OnceCell so that the batch evaluator (if enabled) is globally available.
pub static BATCH_EVALUATOR: OnceCell<BatchEvaluator> = OnceCell::new();

#[doc(hidden)]
thread_local! {
    pub static IS_EVALUATOR: Cell<bool> = Cell::new(false);
}

impl BatchEvaluator {
    pub fn new(num_threads: usize, value: &ValueNetwork, policy: &PolicyNetwork) -> Self {
        let (value_sender, value_receiver) = mpsc::channel::<ValueEvalRequest>();
        let (policy_sender, policy_receiver) = mpsc::channel::<PolicyEvalRequest>();

        let value_receiver = Arc::new(Mutex::new(value_receiver));
        let policy_receiver = Arc::new(Mutex::new(policy_receiver));

        let value_arc = Arc::new(*value);
        let policy_arc = Arc::new(*policy);

        // Spawn evaluator threads for the value network.
        for _ in 0..num_threads {
            let receiver = value_receiver.clone();
            let value_clone = value_arc.clone();
            thread::spawn(move || {
                IS_EVALUATOR.with(|flag| flag.set(true));
                loop {
                    let mut batch = Vec::new();
                    {
                        let mut guard = receiver.lock().unwrap();
                        match guard.recv_timeout(Duration::from_millis(10)) {
                            Ok(req) => batch.push(req),
                            Err(_) => continue,
                        }
                        while let Ok(req) = guard.try_recv() {
                            batch.push(req);
                            if batch.len() >= 16 {
                                break;
                            }
                        }
                    }
                    for req in batch {
                        let result = req.board.get_value_wdl(&value_clone, &req.params);
                        let _ = req.response.send(result);
                    }
                }
            });
        }

        // Spawn evaluator threads for the policy network.
        for _ in 0..num_threads {
            let receiver = policy_receiver.clone();
            let policy_clone = policy_arc.clone();
            thread::spawn(move || {
                IS_EVALUATOR.with(|flag| flag.set(true));
                loop {
                    let mut batch = Vec::new();
                    {
                        let mut guard = receiver.lock().unwrap();
                        match guard.recv_timeout(Duration::from_millis(10)) {
                            Ok(req) => batch.push(req),
                            Err(_) => continue,
                        }
                        while let Ok(req) = guard.try_recv() {
                            batch.push(req);
                            if batch.len() >= 16 {
                                break;
                            }
                        }
                    }
                    for req in batch {
                        let result = req.board.get_policy(req.mov, &req.feats, &policy_clone);
                        let _ = req.response.send(result);
                    }
                }
            });
        }

        BatchEvaluator {
            value_sender,
            policy_sender,
        }
    }

    /// Used by search threads to request a value evaluation.
    pub fn evaluate_value(
        &self,
        board: ChessState,
        params: MctsParams,
        _value: &ValueNetwork,
    ) -> f32 {
        let (tx, rx) = mpsc::channel();
        let req = ValueEvalRequest {
            board,
            params,
            response: tx,
        };
        self.value_sender.send(req).unwrap();
        rx.recv().unwrap()
    }

    /// Used by search threads to request a policy evaluation.
    pub fn evaluate_policy(
        &self,
        board: ChessState,
        mov: Move,
        feats: Accumulator<i16, { POLICY_L1 / 2 }>,
        _policy: &PolicyNetwork,
    ) -> f32 {
        let (tx, rx) = mpsc::channel();
        let req = PolicyEvalRequest {
            board,
            mov,
            feats,
            response: tx,
        };
        self.policy_sender.send(req).unwrap();
        rx.recv().unwrap()
    }
}

pub fn init_batch_evaluator(
    num_evaluator_threads: usize,
    value: &ValueNetwork,
    policy: &PolicyNetwork,
) {
    let evaluator = BatchEvaluator::new(num_evaluator_threads, value, policy);
    let _ = BATCH_EVALUATOR.set(evaluator);
}
