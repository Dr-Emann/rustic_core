//! Rabin chunker benchmarks

use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rustic_core::__bench::rabin_chunks;

const KB: usize = 1024;
const MB: usize = 1024 * KB;
const DATA_SIZE: usize = 32 * MB;
const POLY: u64 = 0x003D_A335_8B4D_C173;

struct ChunkSizes {
    name: &'static str,
    size: usize,
    min_size: usize,
    max_size: usize,
}

const CHUNK_SIZES: [ChunkSizes; 2] = [
    ChunkSizes {
        name: "default",
        size: MB,
        min_size: 512 * KB,
        max_size: 8 * MB,
    },
    ChunkSizes {
        name: "small",
        size: 16 * KB,
        min_size: 4 * KB,
        max_size: 64 * KB,
    },
];

fn chunk_all(data: &[u8], chunk_sizes: &ChunkSizes) {
    let ChunkSizes {
        name: _,
        size,
        min_size,
        max_size,
    } = *chunk_sizes;
    let chunks = rabin_chunks(POLY, size, min_size, max_size, data, data.len()).unwrap();
    for chunk in chunks {
        _ = black_box(chunk.unwrap());
    }
}

fn bench_rabin(c: &mut Criterion) {
    let mut random = vec![0u8; DATA_SIZE];
    StdRng::seed_from_u64(23).fill_bytes(&mut random);
    let zeros = vec![0u8; DATA_SIZE];
    let inputs = [("random", random), ("zeros", zeros)];

    let mut group = c.benchmark_group("rabin");
    _ = group.throughput(Throughput::Bytes(DATA_SIZE as u64));
    _ = group.sample_size(20);
    for (input_name, data) in &inputs {
        for chunk_sizes in &CHUNK_SIZES {
            let id = BenchmarkId::new(*input_name, chunk_sizes.name);
            _ = group.bench_with_input(id, data.as_slice(), |b, data| {
                b.iter(|| chunk_all(data, chunk_sizes));
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_rabin);
criterion_main!(benches);
