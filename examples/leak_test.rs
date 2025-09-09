//
// Copyright (c) 2025 Nathan Fiedler
//
use tzarrays::{CyclicArray, OptimalArray};

fn test_cyclic_array() {
    // create and drop an empty array
    let sut = CyclicArray::<String>::new(0);
    drop(sut);

    let mut sut = CyclicArray::<String>::new(10);
    for _ in 0..7 {
        let value = ulid::Ulid::new().to_string();
        sut.push_back(value);
    }
    drop(sut);

    // create and drop an array that is full
    let mut sut = CyclicArray::<String>::new(10);
    while !sut.is_full() {
        let value = ulid::Ulid::new().to_string();
        sut.push_back(value);
    }
    drop(sut);

    let mut sut = CyclicArray::<String>::new(10);
    // push enough to almost fill the buffer
    for _ in 0..7 {
        let value = ulid::Ulid::new().to_string();
        sut.push_back(value);
    }
    // empty the buffer
    while !sut.is_empty() {
        sut.pop_front();
    }
    // push enough to wrap around to the start of the physical buffer
    for _ in 0..7 {
        let value = ulid::Ulid::new().to_string();
        sut.push_back(value);
    }
    drop(sut);
}

fn test_optimal_array() {
    // add only enough values to allocate one data block
    let mut array: OptimalArray<String> = OptimalArray::new();
    let value = ulid::Ulid::new().to_string();
    array.push(value);

    // add enough values to allocate a few data blocks
    let mut array: OptimalArray<String> = OptimalArray::new();
    for _ in 0..12 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }

    // // push a bunch, pop a few, push more to test Grow/Shrink handling of the
    // // extra empty data block
    // let mut array: OptimalArray<u64> = OptimalArray::new();
    // for value in 0..35 {
    //     array.push(value);
    // }
    // for _ in 0..10 {
    //     array.pop();
    // }
    // for value in 0..12 {
    //     array.push(value);
    // }

    // test pushing many elements then dropping
    let mut array: OptimalArray<String> = OptimalArray::new();
    for _ in 0..30_000 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }

    // test pushing many elements then popping all of them
    let mut array: OptimalArray<String> = OptimalArray::new();
    for _ in 0..16384 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    while !array.is_empty() {
        array.pop();
    }

    // IntoIterator: add exactly one element to test special case
    let mut array: OptimalArray<String> = OptimalArray::new();
    let value = ulid::Ulid::new().to_string();
    array.push(value);
    let _ = array.into_iter();

    // IntoIterator: add enough values to allocate a bunch of data blocks
    let mut array: OptimalArray<String> = OptimalArray::new();
    for _ in 0..512 {
        let value = ulid::Ulid::new().to_string();
        array.push(value);
    }
    // skip enough elements to pass over a few data blocks then drop
    for (index, _) in array.into_iter().skip(96).enumerate() {
        if index == 96 {
            // exit the iterator early intentionally
            break;
        }
    }
}

//
// Create and drop collections and iterators in order to test for memory leaks.
// Must allocate Strings in order to fully test the drop implementation.
//
fn main() {
    println!("starting cyclic array testing...");
    test_cyclic_array();
    println!("completed cyclic array testing");
    println!("starting optimal array testing...");
    test_optimal_array();
    println!("completed optimal array testing");
}
