// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for PQ codebook training (build) time and memory.

use arrow_array::types::Float32Type;
use criterion::{Criterion, criterion_group, criterion_main};

#[cfg(target_os = "linux")]
use pprof::criterion::{Output, PProfProfiler};

use arrow_array::FixedSizeListArray;
use lance_arrow::FixedSizeListArrayExt;
use lance_index::vector::pq::builder::PQBuildParams;
use lance_linalg::distance::DistanceType;
use lance_testing::datagen::generate_random_array_with_seed;

fn bench_pq_build(c: &mut Criterion) {
    // 1M vectors × 128 dims — large enough to show memory/time savings,
    // small enough to run in a reasonable time.
    const NUM_VECTORS: usize = 1_000_000;
    const DIMENSION: usize = 128;
    const SEED: [u8; 32] = [42; 32];

    let data = generate_random_array_with_seed::<Float32Type>(NUM_VECTORS * DIMENSION, SEED);
    let fsl = FixedSizeListArray::try_new_from_values(data, DIMENSION as i32).unwrap();

    let params = PQBuildParams {
        num_sub_vectors: 16,
        num_bits: 8,
        max_iters: 10, // fewer iterations for benchmark speed
        kmeans_redos: 1,
        sample_rate: 256,
        ..Default::default()
    };

    c.bench_function(
        &format!(
            "PQ_build({}x{},m={})",
            NUM_VECTORS, DIMENSION, params.num_sub_vectors
        ),
        |b| {
            b.iter(|| {
                let pq = params.build(&fsl, DistanceType::L2).unwrap();
                criterion::black_box(pq);
            });
        },
    );
}

#[cfg(target_os = "linux")]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_pq_build);

#[cfg(not(target_os = "linux"))]
criterion_group!(
    name=benches;
    config = Criterion::default().significance_level(0.1).sample_size(10);
    targets = bench_pq_build);

criterion_main!(benches);
