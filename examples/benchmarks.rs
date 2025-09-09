//
// Copyright (c) 2025 Nathan Fiedler
//
use tzarrays::OptimalArray;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use std::time::Instant;

fn benchmark_optarray(size: usize) {
    let start = Instant::now();
    let mut coll: OptimalArray<usize> = OptimalArray::new();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("optarray create: {:?}", duration);

    // test random access `size` times; use SmallRng to avoid dominating the
    // running time with random number generation
    let mut rng = SmallRng::seed_from_u64(0);
    let start = Instant::now();
    for _ in 0..size {
        let index = rng.random_range(0..size);
        assert_eq!(coll[index], index);
    }
    let duration = start.elapsed();
    println!("optarray random: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("optarray ordered: {:?}", duration);

    // test popping all elements from the array
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("optarray pop-all: {:?}", duration);
    println!("optarray capacity: {}", coll.capacity());
}

fn benchmark_vector(size: usize) {
    let start = Instant::now();
    let mut coll: Vec<usize> = Vec::new();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("vector create: {:?}", duration);

    // test random access `size` times; use SmallRng to avoid dominating the
    // running time with random number generation
    let mut rng = SmallRng::seed_from_u64(0);
    let start = Instant::now();
    for _ in 0..size {
        let index = rng.random_range(0..size);
        assert_eq!(coll[index], index);
    }
    let duration = start.elapsed();
    println!("vector random: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("vector ordered: {:?}", duration);

    // test popping all elements from the vector
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("vector pop-all: {:?}", duration);
    println!("vector capacity: {}", coll.capacity());
}

fn main() {
    println!("creating OptimalArray...");
    benchmark_optarray(100_000_000);
    println!("creating Vec...");
    benchmark_vector(100_000_000);
}
