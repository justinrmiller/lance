// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark of 4-bit LUT distance table summation (RaBitQ inner loop).

use std::iter::repeat_with;

use criterion::{Criterion, black_box, criterion_group, criterion_main};
use lance_linalg::simd::dist_table::{BATCH_SIZE, sum_4bit_dist_table};
use rand::Rng;

fn bench_sum_4bit_dist_table(c: &mut Criterion) {
    let mut rng = rand::rng();

    // DIM=128, 1-bit RaBitQ: code_len = 128 / (8/1) / 2 * 2 = 16 bytes per vector-batch
    // (each byte holds two 4-bit sub-vector codes)
    for (label, n_vectors, code_len) in [
        ("32vec_dim128", 32_usize, 16_usize),
        ("32vec_dim1536", 32, 96),
        ("16Kvec_dim128", 16_000, 16),
    ] {
        // Round n_vectors up to multiple of BATCH_SIZE
        let n = n_vectors.div_ceil(BATCH_SIZE) * BATCH_SIZE;

        let codes: Vec<u8> = repeat_with(|| rng.random::<u8>())
            .take(n * code_len)
            .collect();

        // dist_table is indexed in parallel with codes: BATCH_SIZE * code_len bytes
        let dist_table: Vec<u8> = repeat_with(|| rng.random::<u8>())
            .take(BATCH_SIZE * code_len)
            .collect();

        let mut dists = vec![0u16; n];

        c.bench_function(&format!("sum_4bit_dist_table/{}", label), |b| {
            b.iter(|| {
                dists.fill(0);
                sum_4bit_dist_table(n, code_len, &codes, &dist_table, &mut dists);
                black_box(&dists);
            })
        });
    }
}

criterion_group!(
    name = benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_sum_4bit_dist_table
);
criterion_main!(benches);
