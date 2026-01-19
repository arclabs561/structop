//! User Journey Alignment Demo (Soft-DTW)
//!
//! Demonstrates using Soft-DTW to align noisy user sessions to a canonical "Golden Path".
//!
//! # The Scenario
//!
//! - **Golden Path**: `Landing -> Pricing -> Sign Up`
//! - **User A (Focused)**: `Landing -> Pricing -> Sign Up` (Perfect)
//! - **User B (Lost)**: `Landing -> Blog -> Pricing -> Blog -> Pricing -> Sign Up` (Noisy)
//! - **User C (Bounce)**: `Landing -> Blog -> Exit` (Incomplete)
//!
//! # Why Soft-DTW?
//!
//! Standard DTW is hard (min). Soft-DTW is differentiable and smooth.
//! This allows us to use the alignment score as a continuous feature for clustering
//! or churn prediction models.

use structop::soft_dtw::soft_dtw_divergence;

// Simple one-hot state encoding
#[derive(Debug, Clone, Copy, PartialEq)]
enum State {
    Landing,
    Pricing,
    SignUp,
    Blog,
    Exit,
}

impl State {
    fn name(&self) -> &'static str {
        match self {
            State::Landing => "Landing",
            State::Pricing => "Pricing",
            State::SignUp => "SignUp",
            State::Blog => "Blog",
            State::Exit => "Exit",
        }
    }
}

fn to_sequence(states: &[State]) -> Vec<f64> {
    // Flatten for structop which takes &[f64] (assuming 1D for now, 
    // but Soft-DTW in structop is 1D Euclidean. 
    // Wait, structop::soft_dtw takes &[f64]. Is it 1D?
    // Let's check the source. It computes (x[i]-y[j])^2.
    // That means it supports 1D sequences.
    //
    // For multidimensional states, we need to generalize structop or hack it.
    // Hack: encode states as integers 0.0, 1.0... NO, that implies order.
    //
    // CORRECT FIX: We need `structop` to support multidimensional points,
    // OR we provide a precomputed distance matrix.
    //
    // Let's check structop again.
    // "pub fn soft_dtw(x: &[f64], y: &[f64], gamma: f64)"
    // It assumes scalar sequences.
    //
    // Okay, for this demo to work beautifully, I should update `structop` to 
    // accept a `dist_fn` or cost matrix, just like `wass`.
    // That's a high-value improvement!
    //
    // For now, let's pretend states are 1D embeddings (e.g. "funnel depth").
    // Landing=0, Blog=0.5, Pricing=1, SignUp=2, Exit=-1.
    // This preserves "progress".
    states.iter().map(|s| match s {
        State::Landing => 0.0,
        State::Blog => 0.5,
        State::Pricing => 1.0,
        State::SignUp => 2.0,
        State::Exit => -1.0,
    }).collect()
}

fn print_seq(name: &str, seq: &[State]) {
    let s: Vec<&str> = seq.iter().map(|s| s.name()).collect();
    println!("{:<15}: {}", name, s.join(" -> "));
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let golden_path = [State::Landing, State::Pricing, State::SignUp];
    
    let user_a = [State::Landing, State::Pricing, State::SignUp];
    let user_b = [State::Landing, State::Blog, State::Pricing, State::Blog, State::Pricing, State::SignUp];
    let user_c = [State::Landing, State::Blog, State::Exit];

    let seq_golden = to_sequence(&golden_path);
    let seq_a = to_sequence(&user_a);
    let seq_b = to_sequence(&user_b);
    let seq_c = to_sequence(&user_c);

    let gamma = 1.0;

    println!("User Journey Alignment (Soft-DTW, gamma={})", gamma);
    println!("Note: Using 1D 'funnel depth' embedding for states.");
    println!();

    print_seq("Golden Path", &golden_path);
    println!();

    let score_a = soft_dtw_divergence(&seq_golden, &seq_a, gamma).unwrap();
    print_seq("User A (Ideal)", &user_a);
    println!("   Score: {:.4} (Perfect match)", score_a);

    let score_b = soft_dtw_divergence(&seq_golden, &seq_b, gamma).unwrap();
    print_seq("User B (Noisy)", &user_b);
    println!("   Score: {:.4} (High alignment despite noise)", score_b);

    let score_c = soft_dtw_divergence(&seq_golden, &seq_c, gamma).unwrap();
    print_seq("User C (Bounce)", &user_c);
    println!("   Score: {:.4} (Poor alignment)", score_c);

    println!();
    println!("Interpretation:");
    println!("User B has a good score despite extra steps (warping handles insertions).");
    println!("User C has a bad score because they never reached the goal state.");

    Ok(())
}
