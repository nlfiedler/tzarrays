//
// Copyright (c) 2025 Nathan Fiedler
//
use tzarrays::CyclicArray;

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

//
// Create and drop collections and iterators in order to test for memory leaks.
// Must allocate Strings in order to fully test the drop implementation.
//
fn main() {
    println!("starting cyclic array testing...");
    test_cyclic_array();
    println!("completed cyclic array testing");
}
