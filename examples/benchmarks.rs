//
// Copyright (c) 2025 Nathan Fiedler
//
use std::time::Instant;
use tzarrays::OptimalArray;

fn benchmark_tzarrays(coll: &mut OptimalArray<usize>, size: usize) {
    let start = Instant::now();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("tzarrays create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("tzarrays ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

    // test popping all elements from the array
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("tzarrays pop-all: {:?}", duration);
    println!("tzarrays capacity: {}", coll.capacity());
}

fn benchmark_vector(size: usize) {
    let start = Instant::now();
    let mut coll: Vec<usize> = Vec::new();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("vector create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("vector ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

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
    let size = 100_000_000;
    println!("creating OptimalArray (r=3)...");
    let mut coll: OptimalArray<usize> = OptimalArray::new();
    benchmark_tzarrays(&mut coll, size);
    println!("creating OptimalArray (r=4)...");
    let mut coll: OptimalArray<usize> = OptimalArray::with_r(4);
    benchmark_tzarrays(&mut coll, size);
    println!("creating Vec...");
    benchmark_vector(size);
}
