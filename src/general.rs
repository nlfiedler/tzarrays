//
// Copyright (c) 2025 Nathan Fiedler
//

//! The generalized implementation of the resizable array which allows for the
//! selection of a desired `r` value at the time of construction.
//!
//! # Memory Usage
//!
//! An empty resizable array is approximately 88 bytes in size, and while
//! holding elements it will have a space overhead on the order of O(rN^1/r) as
//! described in the paper. As elements are added the array will grow by
//! allocating additional data blocks. Likewise, as elements are removed from
//! the end of the array, data blocks will be deallocated as they become empty.
//!
//! # Performance
//!
//! The performance and memory usage of this data structure is complicated,
//! please refer to the original paper for details. In terms of time complexity,
//! most operations are on the order of `O(r)` as they involve looping over the
//! variable number of indices of differently sized data blocks.
//!
//! # Safety
//!
//! Because this data structure is allocating memory, copying bytes using raw
//! pointers, and de-allocating memory as needed, there are many `unsafe` blocks
//! throughout the code.

use super::CyclicArray;
use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::fmt;
use std::ops::{Index, IndexMut};

const USIZE_BITS: u32 = (8 * std::mem::size_of::<usize>()) as u32;

/// Faster version of `base.pow(exponent)` using clz, subtraction,
/// multiplication, and bit shifts.
///
/// Certain values are out of range for the optimal array, so not all values are
/// allowed. For example, a base of 16384 and exponent of 5 is never going to
/// happen for this data structure.
#[inline]
fn fast_power(base: usize, exponent: u32) -> usize {
    1 << (((USIZE_BITS - base.leading_zeros()) - 1) * exponent)
}

/// Optimal resizable array for which the unused space is on the order of
/// O(rN^1/r) plus additional overhead for the indices and cyclic arrays.
///
/// Supports push and pop (and swap/remove) operations only -- insert or remove
/// at other locations is not supported.
///
/// See the paper for the details.
pub struct OptimalArray<T> {
    /// N in the paper (number of elements)
    big_n: usize,
    /// b in the paper, the power of two that yields B, the smallest block size
    /// in terms of elements (see page 13 of the paper)
    little_b: usize,
    /// r in the paper for which the largest blocks will hold B^r elements
    r: usize,
    /// when N increases to upper_limit, a Rebuild(2B) is required
    upper_limit: usize,
    /// when N decreases to lower_limit, a Rebuild(B/2) is required
    lower_limit: usize,
    /// n in the paper, number of occupied data blocks for i ∈ [r - 1], and the
    /// zeroth slot is the number of elements in the last block of size B
    little_n: Vec<usize>,
    /// A in the paper, pointers to the index blocks for i ∈ [r - 1]; first slot
    /// is left unused for convenience (see page 9 of the paper)
    big_a: Vec<CyclicArray<*mut T>>,
}

impl<T> OptimalArray<T> {
    /// Return an empty array with zero capacity and `r` is 3.
    pub fn new() -> Self {
        Self::with_r(3)
    }

    /// Return an empty array with zero capacity and the given `r` value.
    ///
    /// Values of `r` less than 2 are not supported. For higher values of `r`
    /// there will be more levels of block sizes and the larger blocks will be
    /// capable of holding many `B` sized blocks. The result is that the
    /// `Rebuild` operation will occur less frequently.
    pub fn with_r(r: usize) -> Self {
        if r < 2 {
            panic!("r should be >= 2 (is {})", r);
        }
        // start with B equal to 4 (little b = 2) because maybe the paper
        // suggests that on page 12 and 13; pointers are at least that large so
        // it would be absurd for blocks to be smaller than their pointers
        //
        // allocate a single empty index since A[0] is never meant to be used;
        // seemingly this helps with the logic in the combine and split
        // functions
        let mut big_a: Vec<CyclicArray<*mut T>> = Vec::with_capacity(r);
        big_a.push(CyclicArray::<*mut T>::new(0));
        for _ in 1..r {
            // default B = 4, thus 2B = 8, the capacity of the A[i] indices
            big_a.push(CyclicArray::<*mut T>::new(8));
        }
        assert_eq!(big_a.len(), r);
        // little_n is equivalent to A[i].len() except for n[0] which tracks the
        // number of elements of the last block at the end of A[1]
        let little_n: Vec<usize> = vec![0; r];
        Self {
            big_n: 0,
            little_b: 2,
            r,
            // upper limit is B^r and default B is 4; B^r can be represented as
            // 2^(4r/2) which is reduced as follows
            upper_limit: 1 << (2 * r),
            // set lower_limit to 0 to prevent rebuilding below B=4
            lower_limit: 0,
            little_n,
            big_a,
        }
    }

    /// Rebuild the indices and blocks with a new (little) b value.
    fn rebuild(&mut self, new_b: usize) {
        // prepare the raw parts for building the new array
        let one_b: usize = 1 << new_b;
        let two_b = 2 * one_b;
        let mut little_n: Vec<usize> = vec![0; self.r];
        let mut big_a: Vec<CyclicArray<*mut T>> = Vec::with_capacity(self.r);
        big_a.push(CyclicArray::<*mut T>::new(0));
        for _ in 1..self.r {
            big_a.push(CyclicArray::<*mut T>::new(two_b));
        }

        // find the largest index k ∈ [r−1] for which nk > 0
        let mut old_k = self.r - 1;
        while old_k > 0 {
            if self.little_n[old_k] > 0 {
                break;
            }
            old_k -= 1;
        }
        let old_b: usize = 1 << self.little_b;

        // coordinates into the old array that will be advanced as elements are
        // copied into the new array parts
        let mut remaining = self.big_n;
        let mut old_block: usize = 0;
        let mut old_slot: usize = 0;

        // copy from the old array into the new raw parts, starting with the
        // largest block size that can be filled with existing data, and moving
        // to smaller block sizes as less data remains
        for k in (1..=self.r - 1).rev() {
            let block_len = fast_power(one_b, k as u32);
            while remaining >= block_len {
                // allocate block of block_len size
                let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
                unsafe {
                    let ptr = alloc(layout).cast::<T>();
                    if ptr.is_null() {
                        handle_alloc_error(layout);
                    }
                    big_a[k].push_back(ptr);
                }

                // iterate as long as the new block still has capacity
                let mut block_slot: usize = 0;
                while block_slot < block_len {
                    // determine how much can be copied in each iteration
                    let old_block_len = fast_power(old_b, old_k as u32);
                    let copy_len = if old_block_len > block_len {
                        block_len
                    } else {
                        old_block_len
                    };

                    // copy from old block to new block
                    unsafe {
                        let src = self.big_a[old_k][old_block].add(old_slot);
                        let dst = big_a[k][little_n[k]].add(block_slot);
                        std::ptr::copy(src, dst, copy_len);
                    }
                    remaining -= copy_len;
                    block_slot += copy_len;
                    old_slot += copy_len;

                    // the old block has been exhausted, move to the next one
                    if old_slot >= old_block_len {
                        // deallocate old block
                        unsafe {
                            let ptr = self.big_a[old_k][old_block];
                            let layout =
                                Layout::array::<T>(old_block_len).expect("unexpected overflow");
                            dealloc(ptr as *mut u8, layout);
                        }
                        old_block += 1;
                        old_slot = 0;
                        if old_block >= self.little_n[old_k] {
                            // no more old blocks at this level, move on down
                            old_k -= 1;
                            old_block = 0;
                        }
                    }
                }
                // only once the new block has been filled can the index into
                // big_a[k] be increased, otherwise the copying above will be to
                // a non-existent block
                little_n[k] += 1;
            }
        }
        assert_eq!(remaining, 0);

        // transition to the new array layout
        self.little_b = new_b;
        let br = new_b * self.r;
        // ensure lower_limit is set to prevent rebuilding to B=2
        self.lower_limit = if new_b > 2 { 1 << (br - 2 * self.r) } else { 0 };
        self.upper_limit = 1 << br;
        // if any B blocks exist, then n[0] must be B because rebuild is called
        // based on the lower/upper bounds which are always powers of two
        if little_n[1] > 0 {
            little_n[0] = one_b;
        }
        self.little_n = little_n;
        self.big_a = big_a;
    }

    /// Combine small blocks into larger blocks.
    fn combine(&mut self) {
        // find the smallest index k ∈ [r−1] for which nk < 2B
        let one_b = 1 << self.little_b;
        let two_b = 2 * one_b;
        let mut k: usize = 1;
        while k < self.r {
            if self.little_n[k] < two_b {
                break;
            }
            k += 1;
        }
        // if k= ∞: error
        assert!(k < self.r, "invoke rebuild first");

        // for i <- k - 1 downto 1 :
        for i in (1..=(k - 1)).rev() {
            // A[i + 1][ni+1] ← Allocate(Bi+1)
            let new_block_len = fast_power(one_b, (i + 1) as u32);
            let layout = Layout::array::<T>(new_block_len).expect("unexpected overflow");
            let new_block_ptr = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                self.big_a[i + 1].push_back(ptr);
                ptr
            };
            // for j <- 0 to B - 1 :
            let old_block_len = fast_power(one_b, i as u32);
            for j in 0..one_b {
                // Copy(A[i][j], 0, A[i + 1][ni+1], jBi, Bi)
                let src = self.big_a[i].pop_front().expect("programming error");
                let dest_slot = j * old_block_len;
                unsafe {
                    let dst = new_block_ptr.add(dest_slot);
                    std::ptr::copy(src, dst, old_block_len);
                    // Deallocate(A[i][j])
                    let layout = Layout::array::<T>(old_block_len).expect("unexpected overflow");
                    dealloc(src as *mut u8, layout);
                }
                // A[i][j] ← A[i][j + B] // Shift indices
                // -- pop_front() effectively did this already
            }
            // ni ← B
            self.little_n[i] = one_b;
            // ni+1 ← ni+1 + 1
            self.little_n[i + 1] += 1;
        }
    }

    /// Split large blocks into smaller blocks.
    ///
    /// Called when there are no B sized blocks in the array.
    fn split(&mut self) {
        // k ← min{ i ∈ [r−1] | ni > 0 }
        let mut k: usize = 1;
        while k < self.r {
            if self.little_n[k] > 0 {
                break;
            }
            k += 1;
        }
        // if k= ∞: error
        assert!(k < self.r, "invoke rebuild first");

        let one_b: usize = 1 << self.little_b;

        // for i <- k - 1 downto 1 :
        for i in (1..=(k - 1)).rev() {
            // ni+1 ← ni+1 - 1
            self.little_n[i + 1] -= 1;
            let old_block_ptr = self.big_a[i + 1].pop_back().expect("programming error");

            // for j <- 0 to B - 1 :
            for j in 0..one_b {
                // A[i][j] ← Allocate(Bi)
                let block_len = fast_power(one_b, i as u32);
                let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
                let new_block_ptr = unsafe {
                    let ptr = alloc(layout).cast::<T>();
                    if ptr.is_null() {
                        handle_alloc_error(layout);
                    }
                    ptr
                };
                self.big_a[i].push_back(new_block_ptr);
                // need to increment ni, another mistake in the paper?
                self.little_n[i] += 1;

                // Copy(A[i + 1][ni+1], jBi, A[i][nj], 0, Bi)
                unsafe {
                    let src = old_block_ptr.add(j * block_len);
                    std::ptr::copy(src, new_block_ptr, block_len);
                }
            }

            // Deallocate(A[i + 1][ni+1])
            let old_block_len = fast_power(one_b, (i + 1) as u32);
            let layout = Layout::array::<T>(old_block_len).expect("unexpected overflow");
            unsafe { dealloc(old_block_ptr as *mut u8, layout) }
        }
        // n0 ← B  -- another mistake in the paper?
        self.little_n[0] = one_b;
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if a new block is allocated that would exceed `isize::MAX` _bytes_.
    pub fn push(&mut self, value: T) {
        if self.big_n == self.upper_limit {
            self.rebuild(self.little_b + 1);
        }
        let one_b: usize = 1 << self.little_b;
        if self.little_n[1] == 2 * one_b && self.little_n[0] == one_b {
            self.combine();
            // combine does nothing to the n0 block, need to fall through to the
            // next condition to ensure we allocate another B block
        }
        if self.little_n[1] == 0 || self.little_n[0] == one_b {
            // A[1][n1] ← Allocate(B)
            let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
            unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                self.big_a[1].push_back(ptr);
            }
            // n1 ← n1 + 1
            self.little_n[1] += 1;
            // n0 ← 0
            self.little_n[0] = 0;
        }
        // A[1][n1−1][n0] ← a
        let ptr = self.big_a[1][self.little_n[1] - 1];
        unsafe {
            std::ptr::write(ptr.add(self.little_n[0]), value);
        }
        // n0 ← n0 + 1
        self.little_n[0] += 1;
        // N ← N + 1
        self.big_n += 1;
    }

    /// Appends an element if there is sufficient spare capacity, otherwise an
    /// error is returned with the element.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn push_within_capacity(&mut self, value: T) -> Result<(), T> {
        if self.capacity() <= self.big_n {
            Err(value)
        } else {
            self.push(value);
            Ok(())
        }
    }

    /// Decrease the size by one, rebuilding the indices, splitting blocks, or
    /// deallocating empty blocks as necessary.
    fn shrink(&mut self) {
        if self.big_n == self.lower_limit {
            self.rebuild(self.little_b - 1);
        }
        if self.little_n[1] == 0 {
            self.split();
        }

        // n0 ← n0 - 1
        self.little_n[0] -= 1;
        // N ← N - 1
        self.big_n -= 1;

        if self.little_n[0] == 0 {
            let one_b: usize = 1 << self.little_b;
            // Deallocate(A[1][n1-1])
            let ptr = self.big_a[1].pop_back().unwrap();
            let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
            unsafe { dealloc(ptr as *mut u8, layout) }
            // n1 ← n1 - 1
            self.little_n[1] -= 1;
            // n0 ← B -- another mistake in the paper?
            if self.little_n[1] > 0 {
                self.little_n[0] = one_b;
            }
        }
    }

    /// Find the level, block, and slot for the last element.
    ///
    /// # Time complexity
    ///
    /// O(r).
    fn locate_last(&self) -> (usize, usize, usize) {
        // usually the last element will be within the small blocks
        if self.little_n[0] > 0 {
            // last element in the partially filled B-sized block
            (1, self.little_n[1] - 1, self.little_n[0] - 1)
        } else if self.little_n[1] > 0 {
            // last element in the last filled B-sized block
            let one_b: usize = 1 << self.little_b;
            (1, self.little_n[1] - 1, one_b - 1)
        } else {
            // find last data block starting at the smallest level
            let mut k: usize = 2;
            while k < self.r {
                if self.little_n[k] > 0 {
                    let one_b: usize = 1 << self.little_b;
                    let block_size = fast_power(one_b, k as u32);
                    return (k, self.little_n[k] - 1, block_size - 1);
                }
                k += 1;
            }
            unreachable!()
        }
    }

    /// Removes the last element from the array and returns it, or `None` if the
    /// array is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.big_n > 0 {
            // need to copy the value first since shrink() will rearrange the
            // array and possibly deallocate the block containing the element
            // let (level, block, slot) = self.locate(self.big_n - 1);
            let (level, block, slot) = self.locate_last();
            let ptr = self.big_a[level][block];
            let value = unsafe { Some(ptr.add(slot).read()) };
            self.shrink();
            value
        } else {
            None
        }
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns true, or `None`` if the predicate returns `false`` or the vector
    /// is empty (the predicate will not be called in that case).
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        if self.big_n == 0 {
            None
        } else if let Some(last) = self.get_mut(self.big_n - 1) {
            if predicate(last) { self.pop() } else { None }
        } else {
            None
        }
    }

    /// Return the number of elements in the array.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn len(&self) -> usize {
        self.big_n
    }

    /// Returns the total number of elements the resizable array can hold
    /// without reallocating.
    ///
    /// # Time complexity
    ///
    /// O(r).
    pub fn capacity(&self) -> usize {
        let one_b: usize = 1 << self.little_b;
        let mut capacity: usize = 0;
        for k in 1..self.r {
            let block_len = fast_power(one_b, k as u32);
            capacity += block_len * self.little_n[k];
        }
        capacity
    }

    /// Returns true if the array has a length of 0.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn is_empty(&self) -> bool {
        self.big_n == 0
    }

    /// Find the level, block, and slot for the given logical index.
    ///
    /// # Time complexity
    ///
    /// O(r).
    fn locate(&self, index: usize) -> (usize, usize, usize) {
        let one_b: usize = 1 << self.little_b;
        let mut block_size: usize = 0;
        // find index k ∈ [r−1] that contains the data block that contains
        // the element located at logical offset `index`
        let mut k: usize = self.r - 1;
        let mut k_offset: usize = 0;
        while k > 0 {
            let nk = self.little_n[k];
            if nk > 0 {
                block_size = fast_power(one_b, k as u32);
                let k_size = block_size * nk;
                if index < (k_offset + k_size) {
                    break;
                }
                k_offset += k_size;
            }
            k -= 1;
        }
        let k_index = index - k_offset;
        if k == 0 {
            // special-case, index is in last block of A[1]
            (1, self.little_n[1], k_index)
        } else {
            let block = k_index / block_size;
            let slot = k_index % block_size;
            (k, block, slot)
        }
    }

    /// Retrieve a reference to the element at the given offset.
    ///
    /// # Time complexity
    ///
    /// O(r).
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.big_n {
            None
        } else {
            let (level, block, slot) = self.locate(index);
            let ptr = self.big_a[level][block];
            unsafe { (ptr.add(slot)).as_ref() }
        }
    }

    /// Returns a mutable reference to an element.
    ///
    /// # Time complexity
    ///
    /// O(r).
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.big_n {
            None
        } else {
            let (k, block, slot) = self.locate(index);
            let ptr = self.big_a[k][block];
            unsafe { (ptr.add(slot)).as_mut() }
        }
    }

    /// Removes an element from the vector and returns it.
    ///
    /// The removed element is replaced by the last element of the vector.
    ///
    /// This does not preserve ordering of the remaining elements.
    ///
    /// # Time complexity
    ///
    /// O(r).
    pub fn swap_remove(&mut self, index: usize) -> T {
        if index >= self.big_n {
            panic!(
                "swap_remove index (is {index}) should be < len (is {})",
                self.big_n
            );
        }
        // retrieve the value at index before overwriting
        let (level, block, slot) = self.locate(index);
        unsafe {
            let index_ptr = self.big_a[level][block].add(slot);
            let value = index_ptr.read();
            // find the pointer of the last element and copy to index pointer
            let (level, block, slot) = self.locate_last();
            let last_ptr = self.big_a[level][block].add(slot);
            std::ptr::copy(last_ptr, index_ptr, 1);
            self.shrink();
            value
        }
    }

    /// Returns an iterator over the array.
    ///
    /// The iterator yields all items from start to end.
    pub fn iter(&self) -> OptArrayIter<'_, T> {
        OptArrayIter {
            array: self,
            index: 0,
        }
    }

    /// Clears the resizable array, removing and dropping all values and
    /// deallocating all previously allocated blocks.
    ///
    /// # Time complexity
    ///
    /// O(n) if elements are droppable, otherwise O(rN^(1/r))
    pub fn clear(&mut self) {
        use std::ptr::{drop_in_place, slice_from_raw_parts_mut};

        // find the largest allocated blocks
        let mut k: usize = self.r - 1;
        while k > 0 && self.little_n[k] == 0 {
            k -= 1;
        }
        let one_b: usize = 1 << self.little_b;

        if self.big_n > 0 && std::mem::needs_drop::<T>() {
            // drop items and deallocate the data blocks

            // smallest block needs special care
            if self.little_n[0] > 0 {
                let ptr = self.big_a[1].pop_back().unwrap();
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(ptr, self.little_n[0]));
                    let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                    dealloc(ptr as *mut u8, layout);
                }
            }

            // drop all elements in all remaining blocks
            for i in 1..=k {
                let len = fast_power(one_b, i as u32);
                while let Some(ptr) = self.big_a[i].pop_front() {
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(ptr, len));
                        let layout = Layout::array::<T>(len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            }
        } else {
            // elements do not need dropping, simply deallocate data blocks
            for i in 1..=k {
                let len = fast_power(one_b, i as u32);
                while let Some(ptr) = self.big_a[i].pop_front() {
                    unsafe {
                        let layout = Layout::array::<T>(len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            }
        }

        // zero out everything to the initial state
        self.big_n = 0;
        self.little_b = 2;
        self.upper_limit = 1 << (2 * self.r);
        self.lower_limit = 0;
        for idx in 0..self.little_n.len() {
            self.little_n[idx] = 0;
        }
    }
}

impl<T> Default for OptimalArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Display for OptimalArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let vc: Vec<String> = self.little_n.iter().map(|c| format!("{}", c)).collect();
        let counts: String = vc.join(",");
        write!(
            f,
            "OptimalArray(n: {}, l: {}, h: {}, b: {}, r: {}, n: {counts})",
            self.big_n, self.lower_limit, self.upper_limit, self.little_b, self.r
        )
    }
}

impl<T> Drop for OptimalArray<T> {
    fn drop(&mut self) {
        self.clear();
    }
}

impl<T> Index<usize> for OptimalArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let Some(item) = self.get(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
    }
}

impl<T> IndexMut<usize> for OptimalArray<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let Some(item) = self.get_mut(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
    }
}

impl<A> FromIterator<A> for OptimalArray<A> {
    fn from_iter<T: IntoIterator<Item = A>>(iter: T) -> Self {
        let mut arr: OptimalArray<A> = OptimalArray::new();
        for value in iter {
            arr.push(value)
        }
        arr
    }
}

/// Immutable array iterator.
pub struct OptArrayIter<'a, T> {
    array: &'a OptimalArray<T>,
    index: usize,
}

impl<'a, T> Iterator for OptArrayIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        let value = self.array.get(self.index);
        self.index += 1;
        value
    }
}

impl<T> IntoIterator for OptimalArray<T> {
    type Item = T;
    type IntoIter = OptArrayIntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        let mut me = std::mem::ManuallyDrop::new(self);
        let little_n = std::mem::take(&mut me.little_n);
        let big_a = std::mem::take(&mut me.big_a);
        OptArrayIntoIter {
            cursor: 0,
            big_n: me.big_n,
            little_b: me.little_b,
            r: me.r,
            little_n,
            big_a,
        }
    }
}

/// An iterator that moves out of an optimal array.
pub struct OptArrayIntoIter<T> {
    /// offset into the array
    cursor: usize,
    /// N in the paper (number of elements)
    big_n: usize,
    /// b in the paper, the power of two that yields B, the smallest block size
    /// in terms of elements (see page 13 of the paper)
    little_b: usize,
    /// r in the paper for which the largest blocks will hold B^r elements
    r: usize,
    /// n in the paper, number of occupied data blocks for i ∈ [r - 1], and the
    /// zeroth slot is the number of elements in the last block of size B
    little_n: Vec<usize>,
    /// A in the paper, pointers to the index blocks for i ∈ [r - 1]; first slot
    /// is left unused for convenience (see page 9 of the paper)
    big_a: Vec<CyclicArray<*mut T>>,
}

impl<T> OptArrayIntoIter<T> {
    fn locate(&self, index: usize) -> (usize, usize, usize) {
        let one_b: usize = 1 << self.little_b;
        let mut block_size: usize = 0;
        // find index k ∈ [r−1] that contains the data block that contains
        // the element located at logical offset `index`
        let mut k: usize = self.r - 1;
        let mut k_offset: usize = 0;
        while k > 0 {
            let nk = self.little_n[k];
            if nk > 0 {
                block_size = fast_power(one_b, k as u32);
                let k_size = block_size * nk;
                if index < (k_offset + k_size) {
                    break;
                }
                k_offset += k_size;
            }
            k -= 1;
        }
        let k_index = index - k_offset;
        if k == 0 {
            // special-case, index is in last block of A[1]
            (1, self.little_n[1], k_index)
        } else {
            let block = k_index / block_size;
            let slot = k_index % block_size;
            (k, block, slot)
        }
    }
}

impl<T> Iterator for OptArrayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.big_n {
            let (level, block, slot) = self.locate(self.cursor);
            self.cursor += 1;
            unsafe { Some((self.big_a[level][block].add(slot)).read()) }
        } else {
            None
        }
    }
}

impl<T> Drop for OptArrayIntoIter<T> {
    fn drop(&mut self) {
        use std::ptr::{drop_in_place, slice_from_raw_parts_mut};

        // find the largest allocated blocks
        let mut k: usize = self.r - 1;
        while k > 0 && self.little_n[k] == 0 {
            k -= 1;
        }
        let one_b: usize = 1 << self.little_b;

        if std::mem::needs_drop::<T>() {
            // drop items and deallocate the data blocks
            let (first_level, first_block, first_slot) = self.locate(self.cursor);
            let (last_level, last_block, last_slot) = self.locate(self.big_n - 1);
            if first_level == last_level && first_block == last_block {
                // special-case, remaining values are in only one block
                if first_slot <= last_slot {
                    let len = fast_power(one_b, first_level as u32);
                    unsafe {
                        // last_slot is pointing at the last element, need to
                        // add one to include it in the slice
                        let ptr = self.big_a[first_level][first_block];
                        drop_in_place(slice_from_raw_parts_mut(
                            ptr.add(first_slot),
                            last_slot - first_slot + 1,
                        ));
                        let layout = Layout::array::<T>(len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            } else {
                // first partial block needs special care
                let block_len = fast_power(one_b, first_level as u32);
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(
                        self.big_a[first_level][first_block].add(first_slot),
                        block_len - first_slot,
                    ));
                }

                // deallocate all blocks already visited
                for _ in 0..=first_block {
                    let ptr = self.big_a[first_level].pop_front().unwrap();
                    let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
                    unsafe { dealloc(ptr as *mut u8, layout) }
                }

                // smallest block needs special care
                if self.little_n[0] > 0 {
                    let ptr = self.big_a[1].pop_back().unwrap();
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(ptr, self.little_n[0]));
                        let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }

                // drop all elements in all remaining blocks
                for i in (1..=k).rev() {
                    let len = fast_power(one_b, i as u32);
                    while let Some(ptr) = self.big_a[i].pop_front() {
                        unsafe {
                            drop_in_place(slice_from_raw_parts_mut(ptr, len));
                            let layout = Layout::array::<T>(len).expect("unexpected overflow");
                            dealloc(ptr as *mut u8, layout);
                        }
                    }
                }
            }
        } else {
            // no drop, just deallocate the data blocks
            for i in (1..=k).rev() {
                let len = fast_power(one_b, i as u32);
                while let Some(ptr) = self.big_a[i].pop_front() {
                    unsafe {
                        let layout = Layout::array::<T>(len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            }
        }

        // zero out everything to the initial state
        self.cursor = 0;
        self.big_n = 0;
        self.little_b = 2;
        for idx in 0..self.little_n.len() {
            self.little_n[idx] = 0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_power() {
        let bases: Vec<usize> = vec![4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096];
        let exponents: Vec<u32> = vec![0, 1, 2, 3, 4, 5];
        for base in bases.iter() {
            for exp in exponents.iter() {
                assert_eq!(fast_power(*base, *exp), base.pow(*exp));
            }
        }
    }

    #[test]
    fn test_general_array_when_empty() {
        let sut: OptimalArray<usize> = OptimalArray::new();
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);
        assert!(sut.is_empty());
    }

    #[test]
    fn test_general_array_capacity() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        assert_eq!(sut.capacity(), 0);
        sut.push(1);
        assert_eq!(sut.capacity(), 4);
        sut.push(2);
        sut.push(3);
        sut.push(4);
        sut.push(5);
        assert_eq!(sut.capacity(), 8);
        for value in 6..46 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 45);
        assert_eq!(sut.capacity(), 48);
    }

    #[test]
    fn test_general_array_push_within_capacity() {
        // empty array has no allocated space
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        assert_eq!(sut.push_within_capacity(101), Err(101));
        sut.push(1);
        sut.push(2);
        assert_eq!(sut.push_within_capacity(3), Ok(()));
        assert_eq!(sut.push_within_capacity(4), Ok(()));
        assert_eq!(sut.push_within_capacity(5), Err(5));
    }

    #[test]
    fn test_general_array_get_mut_index_mut() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        sut.push(1);
        sut.push(2);
        sut.push(3);
        if let Some(value) = sut.get_mut(1) {
            *value = 11;
        } else {
            panic!("get_mut() returned None")
        }
        assert_eq!(sut[1], 11);
        sut[2] = 22;
        assert_eq!(sut[2], 22);
    }

    #[test]
    fn test_general_array_iter_pop() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..1000 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1000);
        for (index, value) in sut.iter().enumerate() {
            assert_eq!(sut[index], *value);
        }
        for value in (0..1000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
    }

    #[test]
    fn test_general_array_push_many_ints_r2() {
        let mut sut: OptimalArray<usize> = OptimalArray::with_r(2);
        for value in 0..1_000_000 {
            sut.push(value);
        }
        // Rebuild(2B) will have been called 9 times
        assert_eq!(sut.len(), 1_000_000);
        for index in 0..1_000_000 {
            assert_eq!(sut[index], index);
        }
        assert_eq!(sut[99_999], 99_999);
        assert_eq!(sut.capacity(), 1000448);

        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }

        // and do it again to be sure shrinking works correctly
        for value in 0..1_000_000 {
            sut.push(value);
        }
        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
    }

    #[test]
    fn test_general_array_push_many_ints_r3() {
        let mut sut: OptimalArray<usize> = OptimalArray::with_r(3);
        for value in 0..1_000_000 {
            sut.push(value);
        }
        // Rebuild(2B) will have been called 5 times
        assert_eq!(sut.len(), 1_000_000);
        for index in 0..1_000_000 {
            assert_eq!(sut[index], index);
        }
        assert_eq!(sut[99_999], 99_999);
        assert_eq!(sut.capacity(), 1000064);

        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }

        // and do it again to be sure shrinking works correctly
        for value in 0..1_000_000 {
            sut.push(value);
        }
        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
    }

    #[test]
    fn test_general_array_push_many_ints_r4() {
        let mut sut: OptimalArray<usize> = OptimalArray::with_r(4);
        for value in 0..1_000_000 {
            sut.push(value);
        }
        // Rebuild(2B) will have been called 3 times
        assert_eq!(sut.len(), 1_000_000);
        for index in 0..1_000_000 {
            assert_eq!(sut[index], index);
        }
        assert_eq!(sut[99_999], 99_999);
        assert_eq!(sut.capacity(), 1000000);

        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }

        // and do it again to be sure shrinking works correctly
        for value in 0..1_000_000 {
            sut.push(value);
        }
        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
    }

    #[test]
    fn test_general_array_push_many_ints_r5() {
        let mut sut: OptimalArray<usize> = OptimalArray::with_r(5);
        for value in 0..1_000_000 {
            sut.push(value);
        }
        // Rebuild(2B) will have been called 2 times
        assert_eq!(sut.len(), 1_000_000);
        for index in 0..1_000_000 {
            assert_eq!(sut[index], index);
        }
        assert_eq!(sut[99_999], 99_999);
        assert_eq!(sut.capacity(), 1000000);

        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }

        // and do it again to be sure shrinking works correctly
        for value in 0..1_000_000 {
            sut.push(value);
        }
        for value in (0..1_000_000).rev() {
            assert_eq!(sut.pop(), Some(value));
        }
    }

    #[test]
    fn test_general_array_pop_if() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        assert!(sut.pop_if(|_| panic!("should not be called")).is_none());
        for value in 0..10 {
            sut.push(value);
        }
        assert!(sut.pop_if(|_| false).is_none());
        let maybe = sut.pop_if(|v| *v == 9);
        assert_eq!(maybe.unwrap(), 9);
        assert!(sut.pop_if(|v| *v == 9).is_none());
    }

    #[test]
    fn test_general_array_swap_remove_single_block() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        assert_eq!(sut.len(), 1);
        let one = sut.swap_remove(0);
        assert_eq!(one, 1);
        assert_eq!(sut.len(), 0);
    }

    #[test]
    fn test_general_array_swap_remove_multiple_blocks() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        for value in 0..1024 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1024);
        let eighty = sut.swap_remove(80);
        assert_eq!(eighty, 80);
        assert_eq!(sut.pop(), Some(1022));
        assert_eq!(sut[80], 1023);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 0) should be < len (is 0)")]
    fn test_general_array_swap_remove_panic_empty() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.swap_remove(0);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 1) should be < len (is 1)")]
    fn test_general_array_swap_remove_panic_range_edge() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        sut.swap_remove(1);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 2) should be < len (is 1)")]
    fn test_general_array_swap_remove_panic_range_exceed() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        sut.swap_remove(2);
    }

    #[test]
    fn test_general_array_from_iterator() {
        let mut inputs: Vec<i32> = Vec::new();
        for value in 0..10_000 {
            inputs.push(value);
        }
        let sut: OptimalArray<i32> = inputs.into_iter().collect();
        assert_eq!(sut.len(), 10_000);
        for idx in 0..10_000i32 {
            let maybe = sut.get(idx as usize);
            assert!(maybe.is_some(), "{idx} is none");
            let actual = maybe.unwrap();
            assert_eq!(idx, *actual);
        }
    }

    #[test]
    fn test_general_array_clear_and_reuse_tiny() {
        // clear an array that allocated only one block
        let mut sut: OptimalArray<String> = OptimalArray::new();
        sut.push(String::from("one"));
        assert_eq!(sut.len(), 1);
        sut.clear();
        assert_eq!(sut.len(), 0);
        sut.push(String::from("two"));
        assert_eq!(sut.len(), 1);
        // implicitly drop()
    }

    #[test]
    fn test_general_array_clear_and_reuse_ints() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..1024 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1024);
        sut.clear();
        assert_eq!(sut.len(), 0);
        for value in 0..1024 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1024);
        for idx in 0..1024 {
            assert_eq!(sut[idx], idx);
        }
    }

    #[test]
    fn test_general_array_clear_and_reuse_strings() {
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for value in 0..1024 {
            sut.push(format!("{value}"));
        }
        assert_eq!(sut.len(), 1024);
        sut.clear();
        assert_eq!(sut.len(), 0);
        for value in 0..1024 {
            sut.push(format!("{value}"));
        }
        assert_eq!(sut.len(), 1024);
        for value in 0..1024 {
            assert_eq!(sut[value], format!("{value}"));
        }
        // implicitly drop()
    }

    #[test]
    fn test_general_array_push_get_many_instances_ints() {
        // test allocating, filling, and then dropping many instances
        for _ in 0..1_000 {
            let mut sut: OptimalArray<usize> = OptimalArray::new();
            for value in 0..10_000 {
                sut.push(value);
            }
            assert_eq!(sut.len(), 10_000);
        }
    }

    #[test]
    fn test_general_array_push_get_many_instances_strings() {
        // test allocating, filling, and then dropping many instances
        for _ in 0..1_000 {
            let mut sut: OptimalArray<String> = OptimalArray::new();
            for _ in 0..1_000 {
                let value = ulid::Ulid::new().to_string();
                sut.push(value);
            }
            assert_eq!(sut.len(), 1_000);
        }
    }

    #[test]
    fn test_general_array_into_iterator_ints_done() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..1024 {
            sut.push(value);
        }
        for (idx, elem) in sut.into_iter().enumerate() {
            assert_eq!(idx, elem);
        }
        // sut.len(); // error: ownership of sut was moved
    }

    #[test]
    fn test_general_array_into_iterator_drop_tiny_done() {
        // an array that only requires a single block
        let inputs = ["one", "two"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        for (idx, elem) in sut.into_iter().enumerate() {
            assert_eq!(inputs[idx], elem);
        }
        // sut.len(); // error: ownership of sut was moved
    }

    #[test]
    fn test_general_array_into_iterator_drop_tiny_partial() {
        // an array that only requires a single block and only some need to be
        // dropped after partially iterating the values
        let inputs = ["one", "two", "three", "four"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx >= 1 {
                break;
            }
        }
        // implicitly drop()
    }

    #[test]
    fn test_general_array_into_iterator_drop_large_partial() {
        // visit enough to leave the first large block partially done
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx >= 30 {
                break;
            }
        }
        // implicitly drop()
    }

    #[test]
    fn test_general_array_into_iterator_drop_large_more_partial() {
        // visit enough to leave at least one large block completely done
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        for (idx, _) in sut.into_iter().enumerate() {
            if idx >= 96 {
                break;
            }
        }
        // implicitly drop()
    }
}
