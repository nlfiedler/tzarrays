# Change Log

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).
This file follows the convention described at
[Keep a Changelog](http://keepachangelog.com/en/1.0.0/).

## [1.0.2] - 2026-11-22
### Fixed
- Issue #1, memory leak when `r` equals `2`
- Memory leak in simple implementation.
- Memory leaks in general and simple `IntoIterator` implementations.
- Undefined behavior when dropping a zero-capacity `CyclicArray`, which
  aborts the tests on recent Rust releases.
- Undefined behavior from zero-sized allocations when the element type is a
  zero-sized type.
- Simple array could write pushed values into the wrong block, and `clear()`
  could drop uninitialized slots, after popping had emptied a block.
- Double drop of already-yielded elements when dropping an `IntoIterator` of
  either array that had moved past the larger blocks.

## [1.0.1] - 2025-09-24
### Changed
- Improve performance of `pop()` and `swap_remove()` in some cases.

## [1.0.0] - 2025-09-18
### Changed
- Initial release
