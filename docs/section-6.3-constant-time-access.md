# Section 6.3: Constant-Time Access for Any r

Notes on Section 6.3 of **Optimal resizable arrays** by Tarjan and Zwick (2023),
<https://arxiv.org/abs/2211.11009>, which explains how the general
(`O(rN^(1/r))`, `O(N^(1-1/r))`) implementation of Section 6 can support access
in O(1) worst-case time, independent of `r`.

## What is actually O(1)

Only **access and modify** (`get`, `get_mut`, `Index`) become O(1). Grow and
shrink (`push`, `pop`) remain O(r) amortized, as proved in Lemma 6.3.

The value of `r` is still fixed for a given instance. The improvement is that
locating the block level holding item `i` no longer loops over the `r` levels.
All of the per-level block counts are packed into machine words, so prefix
sums of the counts and the search for the right level become a handful of
mask, shift, and bit-scan operations regardless of `r`.

## Restrictions

1. **B is a power of two**, `B = 2^b`. This crate already does this.
2. **`n1` counts only full blocks of size B.** The partially filled block is
   described solely by `n0 < B`. With this change the counts form a (possibly
   redundant) base-B representation of the length:

   ```
   N = n0 + n1·B + n2·B^2 + … + n(r-1)·B^(r-1)
   ```

   By contrast, `general.rs` counts the partial block in `little_n[1]`, which
   is why its `locate` has a `k == 0` special case.
3. **The counts must fit in a machine word.** Each `n_j` can be as large as
   `2B`, so it is split as `n_j = n⁰_j + B·n¹_j` with `0 ≤ n⁰_j < B` and
   `0 ≤ n¹_j ≤ 2`. The digits are packed into two words, `b` bits per level:

   ```
   N⁰ = (n⁰_(r-1), …, n⁰_1, n⁰_0) in base B
   N¹ = (n¹_(r-1), …, n¹_1, n¹_0) in base B
   ```

   This requires `r·b ≤ w`, the word size. With `w = 64` and `B ≥ 4` (so
   `b ≥ 2`), that means `r ≤ 32` at most; in practice it is equivalent to
   requiring `B^r` to fit in a `u64`. This is the standard word-RAM assumption
   and the main hidden restriction.
4. **`msb(x)`, the index of the most significant set bit, takes O(1) time.**
   The paper discusses ways to achieve this (floating-point conversion, the
   Fredman–Willard method). In Rust, `ilog2`/`leading_zeros` and
   `trailing_zeros` compile to single instructions, so this is free. The least
   significant set bit reduces to it via `lsb(x) = msb(x & -x)`.
5. **Grow and shrink must maintain `N⁰` and `N¹`**, including carries when a
   digit changes (for example, when `n0` reaches `B` the block becomes full,
   so `n0` resets to 0 and `n1` increments). Combine and split still loop over
   levels, so these operations remain O(r) amortized.

## The algorithm

### Prefix counts without a loop

Let `N_k` be the number of items in levels below `k`:

```
N_k = n0 + n1·B + … + n(k-1)·B^(k-1)        (N_0 = 0, N_r = N)
```

With the mask `M_k = (1 << k·b) − 1`, which keeps the low `k` digits:

```
N_k = (N⁰ & M_k) + ((N¹ & M_k) << b)
```

The mask discards the digits for levels `k` and above, and adding the two
words sums the remaining digits. The masks can be precomputed. Alternatively,
the `N_k` values can be maintained explicitly.

### Finding the level

The largest blocks hold the lowest indices, so it is convenient to work with
the item's distance from the end of the array:

```
x = (N − 1) − i
```

Item `i` resides in level `k` exactly when `N_k ≤ x < N_(k+1)`. (Level 0 is the
partially filled block of size B.)

Let `ℓ = msb(x) / b`, the position of the most significant nonzero base-B
digit of `x`, so `x ≥ B^ℓ`. Because every count is at most `2B`, the levels
below `ℓ − 1` hold at most

```
N_(ℓ-1) ≤ 2B·(1 + B + … + B^(ℓ-2)) = (2B / (B − 1))·(B^(ℓ-1) − 1) ≤ 3·B^(ℓ-1)
```

items, which is less than `x` when `B ≥ 4`. Therefore `k ≥ ℓ − 1`, leaving only
three possibilities:

| Condition                | Level                                                   |
|--------------------------|---------------------------------------------------------|
| `x < N_ℓ`                | `k = ℓ − 1`                                             |
| `N_ℓ ≤ x < N_(ℓ+1)`      | `k = ℓ`                                                 |
| `N_(ℓ+1) ≤ x`            | `k = lsb((N⁰ \| N¹) & !M_(ℓ+1)) / b`                    |

The third case finds the smallest level above `ℓ` that has any blocks, which
must exist since `x < N`.

### Locating the slot

Level `k` begins at logical index `N − N_(k+1)`. With
`off = i − (N − N_(k+1))`:

```
block = off >> (k·b)       // ⌊off / B^k⌋
slot  = off & M_k          // off mod B^k
```

For `k = 0` the item is at position `i − (N − N_1)` of the single partially
filled block.

### Edge case

When `x = 0` (the last element), `msb(x)` is undefined; in Rust `0.ilog2()`
panics. The paper does not address this. The level is then simply the
smallest non-empty level, `lsb(N⁰ | N¹) / b`.

## Worked example

`r = 3`, `B = 4` (`b = 2`), with 33 elements. In the paper's convention:
`n0 = 1`, `n1 = 4` (full blocks only), `n2 = 1`, so
`N = 1 + 4·4 + 1·16 = 33`.

Splitting each count as `n_j = n⁰_j + B·n¹_j` (note `n1 = 4 = B`, so its high
digit is 1):

| Level j | n_j | n⁰_j | n¹_j |
|---------|-----|------|------|
| 0       | 1   | 1    | 0    |
| 1       | 4   | 0    | 1    |
| 2       | 1   | 1    | 0    |

```
N⁰ = (1, 0, 1) base 4 = 16 + 1 = 17
N¹ = (0, 1, 0) base 4 = 4
```

Prefix counts (`M_1 = 3`, `M_2 = 15`, `M_3 = 63`):

```
N_1 = (17 & 3)  + ((4 & 3)  << 2) = 1 + 0  = 1
N_2 = (17 & 15) + ((4 & 15) << 2) = 1 + 16 = 17
N_3 = (17 & 63) + ((4 & 63) << 2) = 17 + 16 = 33
```

**Item `i = 20`:** `x = 32 − 20 = 12`, `msb(12) = 3`, so `ℓ = 1`. `x` is not
below `N_1 = 1` but is below `N_2 = 17`, so `k = 1`. Then
`off = 20 − (33 − 17) = 4`, `block = 4 >> 2 = 1`, `slot = 4 & 3 = 0`: the
second small block, slot 0.

**Item `i = 10`:** `x = 22`, `msb(22) = 4`, so `ℓ = 2`. `N_2 = 17 ≤ 22 < 33 = N_3`,
so `k = 2`. Then `off = 10 − (33 − 33) = 10`, `block = 10 >> 4 = 0`,
`slot = 10 & 15 = 10`: the large block, slot 10.

## Practical relevance

The paper concludes Section 6.3 by noting that constant access time
independent of `r` "is only of theoretical interest as in practice the value
of `r` used is expected to be very small", and that the simple O(log r) binary
search, or even the naive O(r) search, "would probably work best".

The technique matters mainly for Corollary 6.2, where `r = log N` gives
`N + O(log N)` space with O(1) access, and an O(r) lookup would be a real cost.

For this crate, `simple::OptimalArray` already has O(1) access because `r` is
fixed at 3. Only `general::OptimalArray` would be affected, and for the small
values of `r` it is likely to be used with, the gain over the current loop in
`locate` is probably small. Benchmarking would settle it.
