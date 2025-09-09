# Optimal resizable arrays

## Overview

This Rust crate implements a resizable array as described in the paper **Optimal resizable arrays** by Tarjan and Zwick, published in 2023.

* https://doi.org/10.48550/arXiv.2211.11009

This data structure supports `push` and `pop` operations and does _not_ support inserts or removes at other locations within the array. One exception is the `swap/remove` operation which will retrieve a value from a specified index, overwrite that slot with the value at the end of the array, decrement the count, and return the retrieved value.

This is part of a collection of similar data structures:

* [Optimal Arrays](https://github.com/nlfiedler/optarray)
    - O(√N) space overhead and O(1) running time for most operations
* [Extensible Arrays](https://github.com/nlfiedler/extarray)
    - O(√N) space overhead and O(1) running time for most operations
* [Segment Array](https://github.com/nlfiedler/segarray)
    - Grows geometrically like `Vec`, hence O(N) space overhead

### Memory Usage

Compared to the `Vec` type in the Rust standard library, this data structure will have substantially less unused space. As an example, with `r` equal to `4` and an array with 1 million entries, at most 32 slots will be unused, while a `Vec` will have 48,576 unused slots (`Vec` capacity after zero starts at 4 and doubles each time). The index blocks contribute to the overhead of this data structure and that is on the order of O(rN^1/r). This data structure will grow and shrink as needed. That is, as `push()` is called, new data blocks will be allocated to contain the new elements. Meanwhile, `pop()` will deallocate data blocks as they become empty. See the paper for a detailed analysis of the space overhead and time complexity.

## Examples

A simple example copied from the unit tests.

```rust
let inputs = [
    "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
];
let mut arr: OptimalArray<String> = OptimalArray::new();
for item in inputs {
    arr.push(item.to_owned());
}
for (idx, elem) in arr.iter().enumerate() {
    assert_eq!(inputs[idx], elem);
}
```

## Supported Rust Versions

The Rust edition is set to `2024` and hence version `1.85.0` is the minimum supported version.

## Troubleshooting

### Memory Leaks

Finding memory leaks with [Address Sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) is fairly [easy](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html) and seems to work best on Linux. The shell script below gives a quick demonstration of running one of the examples with ASAN analysis enabled.

```shell
#!/bin/sh
env RUSTDOCFLAGS=-Zsanitizer=address RUSTFLAGS=-Zsanitizer=address \
    cargo run -Zbuild-std --target x86_64-unknown-linux-gnu --release --example leak_test
```

## References

* \[1\]: [Optimal resizable arrays (2023)](https://arxiv.org/abs/2211.11009)
