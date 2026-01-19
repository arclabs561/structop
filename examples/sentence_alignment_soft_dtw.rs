//! Sentence alignment demo (Soft-DTW with cost matrix).
//!
//! Realistic-ish use case:
//! - You have a clean sentence sequence (reference).
//! - You have a noisy OCR/scrape version with extra boilerplate sentences.
//! - You want an **ordered alignment score** (sequence-aware), not a bag-of-words.
//!
//! This example uses:
//! - cheap char n-gram hashing to embed sentences into vectors
//! - cosine distance to build a cost matrix
//! - `structop::soft_dtw_cost` to compute Soft-DTW on that cost

use ndarray::Array1;

fn embed(text: &str, dim: usize) -> Array1<f32> {
    let mut v = Array1::<f32>::zeros(dim);
    let s = text.to_lowercase();
    let chars: Vec<char> = s.chars().collect();
    if chars.len() >= 3 {
        for i in 0..chars.len() - 2 {
            let h = (chars[i] as usize * 31 * 31
                + chars[i + 1] as usize * 31
                + chars[i + 2] as usize)
                % dim;
            v[h] += 1.0;
        }
    } else {
        for c in chars {
            v[(c as usize) % dim] += 1.0;
        }
    }
    let norm = v.dot(&v).sqrt();
    if norm > 0.0 {
        v /= norm;
    }
    v
}

fn cosine_dist(a: &Array1<f32>, b: &Array1<f32>) -> f64 {
    let dot = a.dot(b);
    ((1.0 - dot).max(0.0)).sqrt() as f64
}

fn split_sentences(text: &str) -> Vec<String> {
    // Very small heuristic splitter: enough for a demo without external deps.
    let mut out = Vec::new();
    let mut cur = String::new();
    for ch in text.chars() {
        cur.push(ch);
        if matches!(ch, '.' | '!' | '?') {
            let s = cur.trim().to_string();
            if !s.is_empty() {
                out.push(s);
            }
            cur.clear();
        }
    }
    let tail = cur.trim().to_string();
    if !tail.is_empty() {
        out.push(tail);
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ref_text = "Quarterly earnings showed steady growth in all sectors. Revenue was up 12% year-over-year. Guidance remains unchanged.";
    let noisy_text = "CONFIDENTIAL - INTERNAL MEMO. Qarterly earnigns showd stdy grwth across sectrs. Revnue +12 percent YoY. MENU HOME CONTACT. Guidance remains unchngd.";

    let ref_sents = split_sentences(ref_text);
    let noisy_sents = split_sentences(noisy_text);

    println!("Reference sentences ({}):", ref_sents.len());
    for (i, s) in ref_sents.iter().enumerate() {
        println!("  {i}: {s}");
    }
    println!();
    println!("Noisy sentences ({}):", noisy_sents.len());
    for (i, s) in noisy_sents.iter().enumerate() {
        println!("  {i}: {s}");
    }
    println!();

    let dim = 256;
    let ref_vecs: Vec<Array1<f32>> = ref_sents.iter().map(|s| embed(s, dim)).collect();
    let noisy_vecs: Vec<Array1<f32>> = noisy_sents.iter().map(|s| embed(s, dim)).collect();

    let n = ref_vecs.len();
    let m = noisy_vecs.len();
    let mut cost = vec![0.0f64; n * m];
    for i in 0..n {
        for j in 0..m {
            cost[i * m + j] = cosine_dist(&ref_vecs[i], &noisy_vecs[j]);
        }
    }

    let gamma = 0.5;
    let sdtw = structop::soft_dtw::soft_dtw_cost(&cost, n, m, gamma)?;

    println!("Soft-DTW value (gamma={gamma}): {sdtw:.6}");
    println!();

    // For interpretability, show greedy best matches by cost (not the full DTW path).
    println!("Greedy best sentence matches (by min cost):");
    for i in 0..n {
        let mut best_j = 0usize;
        let mut best = f64::INFINITY;
        for j in 0..m {
            let d = cost[i * m + j];
            if d < best {
                best = d;
                best_j = j;
            }
        }
        println!("  ref[{i}] -> noisy[{best_j}]  dist={best:.3}");
        println!("    ref : {}", ref_sents[i]);
        println!("    noisy: {}", noisy_sents[best_j]);
    }

    Ok(())
}

