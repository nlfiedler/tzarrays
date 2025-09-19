//
// Copyright (c) 2025 Nathan Fiedler
//
use std::time::Instant;
use tzarrays::general::OptimalArray as GeneralArray;
use tzarrays::simple::OptimalArray as SimpleArray;

fn benchmark_general(coll: &mut GeneralArray<usize>, size: usize) {
    let start = Instant::now();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("general create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("general ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

    // test popping all elements from the array
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("general pop-all: {:?}", duration);
    println!("general capacity: {}", coll.capacity());
}

fn benchmark_simple(coll: &mut SimpleArray<usize>, size: usize) {
    let start = Instant::now();
    for value in 0..size {
        coll.push(value);
    }
    let duration = start.elapsed();
    println!("simple create: {:?}", duration);

    // test sequenced access for entire collection
    let start = Instant::now();
    for (index, value) in coll.iter().enumerate() {
        assert_eq!(*value, index);
    }
    let duration = start.elapsed();
    println!("simple ordered: {:?}", duration);

    let unused = coll.capacity() - coll.len();
    println!("unused capacity: {unused}");

    // test popping all elements from the array
    let start = Instant::now();
    while !coll.is_empty() {
        coll.pop();
    }
    let duration = start.elapsed();
    println!("simple pop-all: {:?}", duration);
    println!("simple capacity: {}", coll.capacity());
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
    println!("creating GeneralArray (r=3)...");
    let mut coll: GeneralArray<usize> = GeneralArray::new();
    benchmark_general(&mut coll, size);
    println!("creating GeneralArray (r=4)...");
    let mut coll: GeneralArray<usize> = GeneralArray::with_r(4);
    benchmark_general(&mut coll, size);
    println!("creating SimpleArray...");
    let mut coll: SimpleArray<usize> = SimpleArray::new();
    benchmark_simple(&mut coll, size);
    println!("creating Vec...");
    benchmark_vector(size);
}
