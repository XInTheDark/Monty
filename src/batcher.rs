use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use crate::chess::ChessState;
use crate::networks::ValueNetwork;
use crate::mcts::MctsParams;

/// A request sent to the batch evaluator.
pub struct EvalRequest {
    /// The board that needs to be evaluated.
    pub board: ChessState,
    /// A one‐shot sender to deliver the evaluation result.
    pub responder: mpsc::Sender<f32>,
}

/// The batcher collects evaluation requests from tree–search threads and
/// processes them in a batch (over a short timeout or after a batch size is reached).
pub struct Batcher {
    sender: mpsc::Sender<EvalRequest>,
    receiver: Arc<Mutex<mpsc::Receiver<EvalRequest>>>,
    value: Arc<ValueNetwork>,
    params: Arc<MctsParams>,
    /// Maximum number of requests to process at once.
    pub batch_size: usize,
    /// Maximum time to wait before processing a non–full batch.
    pub batch_timeout: Duration,
}

impl Batcher {
    /// Create a new batcher given shared references (Arc) to the value network and the MCTS parameters.
    pub fn new(value: Arc<ValueNetwork>, params: Arc<MctsParams>) -> Self {
        let (sender, receiver) = mpsc::channel();
        Batcher {
            sender,
            receiver: Arc::new(Mutex::new(receiver)),
            value,
            params,
            batch_size: 8, // set desired batch size (you can tune this)
            batch_timeout: Duration::from_millis(10),
        }
    }

    /// Called by a search thread when it needs to evaluate a leaf.
    /// The board is cloned, a request is sent and the thread waits for its result.
    pub fn evaluate_value(&self, board: &ChessState) -> f32 {
        let (resp_tx, resp_rx) = mpsc::channel();
        let req = EvalRequest {
            board: board.clone(),
            responder: resp_tx,
        };
        // Send the evaluation request (ignore errors).
        let _ = self.sender.send(req);
        // Wait for the result (default to 0.5 if something goes wrong)
        resp_rx.recv().unwrap_or(0.5)
    }

    /// This is the main loop run by each evaluator thread.
    /// It collects evaluation requests in a batch (either until batch_size is reached or
    /// batch_timeout expires) and then processes them one‐by‐one (the “batch” can then be replaced
    /// by a vectorized call if desired).
    pub fn run_evaluator_loop(&self) {
        loop {
            let mut batch = Vec::with_capacity(self.batch_size);
            let start = Instant::now();
            {
                // Lock the receiver and try to collect up to batch_size requests.
                let rx_lock = self.receiver.lock().unwrap();
                while batch.len() < self.batch_size {
                    let elapsed = start.elapsed();
                    if elapsed >= self.batch_timeout {
                        break;
                    }
                    match rx_lock.recv_timeout(self.batch_timeout - elapsed) {
                        Ok(req) => batch.push(req),
                        Err(mpsc::RecvTimeoutError::Timeout) => break,
                        Err(mpsc::RecvTimeoutError::Disconnected) => return,
                    }
                }
            }
            if batch.is_empty() {
                // Sleep briefly to avoid a busy loop.
                std::thread::sleep(Duration::from_millis(1));
                continue;
            }
            // Process the batch. Here we simply evaluate each board via get_value_wdl.
            // (In future this loop might be vectorized.)
            let mut results = Vec::with_capacity(batch.len());
            for req in &batch {
                let eval = req.board.get_value_wdl(&self.value, &self.params);
                results.push(eval);
            }
            // Return results to each waiting thread.
            for (req, result) in batch.into_iter().zip(results.into_iter()) {
                let _ = req.responder.send(result);
            }
        }
    }
}
