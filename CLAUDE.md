# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Rust library crate (edition 2024, MSRV 1.85) implementing the resizable array from **Optimal resizable arrays** by Tarjan & Zwick (2023, https://arxiv.org/abs/2211.11009). No runtime dependencies; `ulid` is a dev-dependency used to generate `String` test values. Sibling crates by the same author (`optarray`, `extarray`, `segarray`) implement similar structures with a comparable API.

## Commands

```shell
cargo build
cargo test                                   # all unit tests (they live in #[cfg(test)] mod tests in each src file)
cargo test test_simple_array_push_many_ints  # single test by name (substring match)
cargo test general::                         # tests in one module
cargo run --release --example benchmarks     # timing comparison of simple vs general
cargo run --example leak_test                # exercises drop paths; meant to be run under ASan
```

Memory-leak checking uses AddressSanitizer in a Linux/nightly Docker container (`containers/`):

```shell
cd containers
docker compose build --pull
docker compose run leaktest
sh leak.sh          # runs leak_test example and cargo test with -Zsanitizer=address
exit
docker compose down
```

The Dockerfile copies only `Cargo.toml`, `src`, and `examples`, so rebuild the image after source changes.

## Architecture

Three public types, all built on raw `alloc`/`dealloc` and `unsafe` pointer copies (no `Vec` for element storage):

- `CyclicArray<T>` (`src/lib.rs`) — fixed-capacity circular buffer; panics on push when full. Used internally as the *index* of data-block pointers (`CyclicArray<*mut T>`), which is why blocks can be removed from the front cheaply during `combine`.
- `general::OptimalArray<T>` (`src/general.rs`) — the full algorithm with configurable `r >= 2` (`with_r`, default 3). Data blocks have sizes `B, B^2, …, B^(r-1)`. `big_a[i]` is the index of blocks of size `B^i` (`big_a[0]` is intentionally an empty, unused placeholder to match the paper's 1-based indexing); `little_n[i]` is the block count at level `i`, except `little_n[0]`, which is the number of elements in the last (only possibly partial) `B`-sized block. Most operations loop over `r`.
- `simple::OptimalArray<T>` (`src/simple.rs`) — Section 5 of the paper: `r` fixed at 3, with explicit `small` (size `B`, index capacity `2B`) and `large` (size `B^2`, index capacity `B`) indices and fields `n0/n1/n2`. Avoids loops, so it's notably faster. It also keeps at most one empty `B` block (`empty`) to avoid dealloc/alloc churn on alternating pop/push.

Shared mechanics (both implementations):
- `B = 2^little_b`, starting at `B = 4`. When `N` reaches `upper_limit` (`B^r`) the structure is rebuilt with `2B`; when it drops to `lower_limit` it's rebuilt with `B/2` (`lower_limit = 0` while `B = 4`, so it never shrinks below that).
- `push` calls `combine` when the `B`-level index is full (merges `B` blocks into one block of the next size up); `pop`/`shrink` calls `split` when no `B` blocks remain. Code comments quote the paper's pseudocode line-by-line and note places where the paper appears to be wrong.
- `locate(index)` maps a logical index to (level, block, slot). `IntoIterator` has its own copy of the block state and its own `locate`, and its `Drop` must free remaining elements and blocks — leaks have historically shown up in `Drop`/`IntoIter`/`clear` paths, so changes there should be checked with the leak test.
- Only `push`/`pop`/`pop_if`/`swap_remove` mutate length; arbitrary insert/remove is deliberately unsupported.

`TODO.org` holds design notes and analysis of the paper (notation, Figure 4 field meanings, example `B` values).

## Conventions

When adding behavior, tests are written for both `simple` and `general` (and for `general`, typically for several `r` values, e.g. 2–5), using both `usize` and `String` element types so drop behavior is exercised. Add corresponding cases to `examples/leak_test.rs` for new allocation/drop paths.
