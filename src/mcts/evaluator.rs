use std::collections::VecDeque;
use std::sync::{Condvar, Mutex};
use std::sync::mpsc::{channel, Sender};
use std::thread;
use once_cell::sync::Lazy;
use std::time::Duration;

use crate::chess::ChessState;
use crate::mcts::MctsParams;
use crate::networks::{PolicyNetwork, ValueNetwork};
use crate::networks::accumulator::Accumulator;
use crate::networks::POLICY_L1;

#[derive(Copy, Clone)]
pub struct SharedPtr<T>(*const T);

impl<T> SharedPtr<T> {
    pub fn new(ptr: *const T) -> Self {
        SharedPtr(ptr)
    }
    pub fn get(&self) -> *const T {
        self.0
    }
}

unsafe impl<T: Sync> Send for SharedPtr<T> {}
unsafe impl<T: Sync> Sync for SharedPtr<T> {}

pub enum EvalJob {
    EvaluateValue {
        state: ChessState,
        params: MctsParams,
        value: SharedPtr<ValueNetwork>,
        ret: Sender<f32>,
    },
    EvaluatePolicy {
        state: ChessState,
        mov: crate::chess::Move,
        feats: Accumulator<i16, { POLICY_L1 / 2 }>,
        policy: SharedPtr<PolicyNetwork>,
        ret: Sender<f32>,
    },
}

struct EvalQueue {
    queue: Mutex<VecDeque<EvalJob>>,
    condvar: Condvar,
    shutdown: Mutex<bool>,
}

impl EvalQueue {
    fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
            condvar: Condvar::new(),
            shutdown: Mutex::new(false),
        }
    }
}

static EVAL_QUEUE: Lazy<EvalQueue> = Lazy::new(|| EvalQueue::new());

static mut EVALUATOR_HANDLES: Option<Mutex<Vec<thread::JoinHandle<()>>>> = None;

pub fn setup(num_threads: usize) {
    let mut handles = Vec::new();
    for _ in 0..num_threads {
        let handle = thread::spawn(|| {
            evaluator_thread();
        });
        handles.push(handle);
    }
    unsafe {
        EVALUATOR_HANDLES = Some(Mutex::new(handles));
    }
}

pub fn shutdown() {
    {
        let mut shut = EVAL_QUEUE.shutdown.lock().unwrap();
        *shut = true;
        EVAL_QUEUE.condvar.notify_all();
    }
    unsafe {
        if let Some(ref mutex_handles) = EVALUATOR_HANDLES {
            let mut handles = mutex_handles.lock().unwrap();
            for handle in handles.drain(..) {
                let _ = handle.join();
            }
        }
        EVALUATOR_HANDLES = None;
    }
}

fn evaluator_thread() {
    loop {
        // Lock the queue.
        let mut queue_guard = EVAL_QUEUE.queue.lock().unwrap();
        while queue_guard.is_empty() && !*EVAL_QUEUE.shutdown.lock().unwrap() {
            queue_guard = EVAL_QUEUE.condvar.wait(queue_guard).unwrap();
        }
        if *EVAL_QUEUE.shutdown.lock().unwrap() && queue_guard.is_empty() {
            break;
        }
        // Gather a batch (up to 16 jobs).
        let mut batch = Vec::new();
        while let Some(job) = queue_guard.pop_front() {
            batch.push(job);
            if batch.len() >= 16 {
                break;
            }
        }
        drop(queue_guard);
        // Process each job in the batch.
        for job in batch {
            match job {
                EvalJob::EvaluateValue { state, params, value, ret } => {
                    let val_net: &ValueNetwork = unsafe { &*value.get() };
                    let result = state.get_value_wdl(val_net, &params);
                    let _ = ret.send(result);
                }
                EvalJob::EvaluatePolicy { state, mov, feats, policy, ret } => {
                    let pol_net: &PolicyNetwork = unsafe { &*policy.get() };
                    let result = state.get_policy(mov, &feats, pol_net);
                    let _ = ret.send(result);
                }
            }
        }
    }
}

/// Public API for value evaluation. If the evaluator pool is set up the job is queued;
/// otherwise the evaluation is performed directly.
pub fn evaluate_value(state: &ChessState, value: &ValueNetwork, params: &MctsParams) -> f32 {
    unsafe {
        if EVALUATOR_HANDLES.is_some() {
            let (tx, rx) = channel();
            let job = EvalJob::EvaluateValue {
                state: state.clone(),
                params: params.clone(),
                value: SharedPtr::new(value as *const ValueNetwork),
                ret: tx,
            };
            {
                let mut queue = EVAL_QUEUE.queue.lock().unwrap();
                queue.push_back(job);
                EVAL_QUEUE.condvar.notify_one();
            }
            rx.recv().unwrap_or_else(|_| state.get_value_wdl(value, params))
        } else {
            state.get_value_wdl(value, params)
        }
    }
}

/// Public API for policy evaluation.
pub fn evaluate_policy(state: &ChessState, mov: crate::chess::Move, feats: &Accumulator<i16, { POLICY_L1 / 2 }>, policy: &PolicyNetwork) -> f32 {
    unsafe {
        if EVALUATOR_HANDLES.is_some() {
            let (tx, rx) = channel();
            let job = EvalJob::EvaluatePolicy {
                state: state.clone(),
                mov,
                feats: feats.clone(),
                policy: SharedPtr::new(policy as *const PolicyNetwork),
                ret: tx,
            };
            {
                let mut queue = EVAL_QUEUE.queue.lock().unwrap();
                queue.push_back(job);
                EVAL_QUEUE.condvar.notify_one();
            }
            rx.recv().unwrap_or_else(|_| state.get_policy(mov, feats, policy))
        } else {
            state.get_policy(mov, feats, policy)
        }
    }
}
