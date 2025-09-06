//
// Copyright (c) 2025 Nathan Fiedler
//
#![allow(dead_code)]
#![allow(unused_imports)]
#![allow(unused_variables)]

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::fmt;
use std::mem::MaybeUninit;
use std::ops::Index;
use std::ptr::null_mut;

pub struct OptimalArray<T> {
    /// N in the paper (number of elements)
    big_n: usize,
    /// b in the paper, the power of two that yields B, the smallest block size
    /// in terms of elements (see page 13 of the paper)
    little_b: usize,
    /// r in the paper for which the largest blocks will hold B^r elements
    r: usize,
    /// when n increases to upper_limit, a Rebuild(2B) is required
    upper_limit: usize,
    /// when n decreases to lower_limit, a Rebuild(B/2) is required
    lower_limit: usize,
    /// n in the paper, number of _full_ data blocks for i ∈ [r - 1], and the
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
    pub fn with_r(r: usize) -> Self {
        if r < 2 {
            panic!("r should be >= 2 (is {})", r);
        }
        // start with B equal to 4 (little b = 2) because maybe the paper
        // suggests that on page 12; pointers are at least that large so it
        // would be absurd for blocks to be smaller than the pointers that point
        // to the blocks
        //
        // allocate a single empty vector in the indices since A[0] is never
        // meant to be used; seemingly this helps with the logic in the combine
        // and split functions
        let mut big_a: Vec<CyclicArray<*mut T>> = Vec::with_capacity(r);
        big_a.push(CyclicArray::<*mut T>::new(0));
        for _ in 1..r {
            // default B = 4, thus 2B = 8, the capacity of the A[i] indices
            big_a.push(CyclicArray::<*mut T>::new(8));
        }
        assert_eq!(big_a.len(), r);
        // little_n is mostly equivalent to A[i].len() except for n[1] which
        // only tracks full blocks in A[1] while n[0] tracks the length of the
        // empty or partial block at the end of A[1]
        let little_n: Vec<usize> = vec![0; r];
        Self {
            big_n: 0,
            little_b: 2,
            r,
            // upper limit is B^r and default B is 4; B^r can be represented as
            // 2^(4r/2) which is reduced as follows
            upper_limit: 1 << (2 * r),
            lower_limit: 1,
            little_n,
            big_a,
        }
    }

    /// Rebuild the indices and blocks with a new (little) b value.
    fn rebuild(&mut self, new_b: usize) {
        // println!("rebuild start: {self} to new_b = {new_b}");
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
        // println!("new_b: {one_b} old_k: {old_k}, old_b: {old_b}");

        // coordinates into the old array that will be advanced as elements are
        // copied into the new array parts
        let mut remaining = self.big_n;
        let mut old_block: usize = 0;
        let mut old_slot: usize = 0;

        // copy from the old array into the new raw parts, starting with the
        // largest block size that can be filled with existing data, and moving
        // to smaller block sizes as less data remains
        for k in (1..=self.r - 1).rev() {
            let block_len = one_b.pow(k as u32);
            // println!("for k = {k}, block_len: {block_len}, remaining: {remaining}");
            while remaining >= block_len {
                // allocate block of block_len size
                // println!("alloc {block_len} in k {k}");
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
                    let old_block_len = old_b.pow(old_k as u32);
                    let copy_len = if old_block_len > block_len {
                        block_len
                    } else {
                        old_block_len
                    };

                    // copy from old block to new block
                    // println!(
                    //     "new block: {block_len:5}, old block: {old_block_len:5}, copy {copy_len:5} : {old_k:2} {old_block:3} {old_slot:5}  > {k:2} {:3} {block_slot:5}",
                    //     little_n[k],
                    // );
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
                        // println!("dealloc {old_block_len} in self.big_a[{old_k}][{old_block}]");
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

        // transition to the new array layout
        self.little_b = new_b;
        let br = new_b * self.r;
        self.lower_limit = 1 << (br - 2 * self.r);
        self.upper_limit = 1 << br;
        self.little_n = little_n;
        self.big_a = big_a;
        // println!("rebuild done: {self}");
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
        // ran out of A indices, rebuild() should have been called
        assert!(k < self.r, "programming error");

        // for i <- k - 1 downto 1 :
        for i in (1..=(k - 1)).rev() {
            // A[i + 1][ni+1] ← Allocate(Bi+1)
            let new_block_len = one_b.pow((i + 1) as u32);
            let layout = Layout::array::<T>(new_block_len).expect("unexpected overflow");
            unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                self.big_a[i + 1].push_back(ptr);
            }
            let new_block_ptr = self.big_a[i + 1][self.little_n[i + 1]];
            // for j <- 0 to B - 1 :
            let old_block_len = one_b.pow(i as u32);
            for block in 0..one_b {
                // Copy(A[i][j], 0, A[i + 1][ni+1], jBi, Bi)
                let src = self.big_a[i].pop_front().expect("programming error");
                let dest_slot = block * old_block_len;
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
    fn split(&mut self) {
        todo!()
    }

    /// Appends an element to the back of a collection.
    ///
    /// # Panics
    ///
    /// Panics if a new block is allocated that would exceed `isize::MAX` _bytes_.
    pub fn push(&mut self, value: T) {
        let mut one_b: usize = 1 << self.little_b;
        if self.big_n == self.upper_limit {
            self.rebuild(self.little_b + 1);
            // rebuild changes b, need to update our local copy
            one_b = 1 << self.little_b;
        }
        if self.little_n[1] == 2 * one_b && self.little_n[0] == one_b {
            self.combine();
            // combine does nothing to the n0 block, need to fall through to the
            // next condition to ensure we allocate another B block; the paper
            // makes the mistake of using if/else-if/else-if when it should be a
            // series of if blocks
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
        todo!()
    }

    /// Decrease the size by one, rebuilding the indices, splitting blocks, or
    /// deallocating empty blocks as necessary.
    fn shrink(&mut self) {
        todo!()
    }

    /// Removes the last element from an array and returns it, or `None` if it
    /// is empty.
    pub fn pop(&mut self) -> Option<T> {
        todo!()
    }

    /// Removes and returns the last element from a vector if the predicate
    /// returns true, or None if the predicate returns false or the vector is
    /// empty (the predicate will not be called in that case).
    pub fn pop_if(&mut self, predicate: impl FnOnce(&mut T) -> bool) -> Option<T> {
        todo!()
    }

    /// Return the number of elements in the array.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn len(&self) -> usize {
        self.big_n
    }

    /// Returns the total number of elements the extensible array can hold
    /// without reallocating.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn capacity(&self) -> usize {
        // it is whatever empty space exists in the last B-sized block
        todo!()
    }

    /// Returns true if the array has a length of 0.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn is_empty(&self) -> bool {
        self.big_n == 0
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
            let one_b: usize = 1 << self.little_b;
            let mut block_size: usize = 0;
            // find index k ∈ [r−1] that contains the data block that contains
            // the element located at logical offset `index`
            let mut k: usize = self.r - 1;
            let mut k_offset: usize = 0;
            while k > 0 {
                let nk = self.little_n[k];
                if nk > 0 {
                    block_size = one_b.pow(k as u32);
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
                let ptr = self.big_a[1][self.little_n[1]];
                unsafe { (ptr.add(k_index)).as_ref() }
            } else {
                let block = k_index / block_size;
                let slot = k_index % block_size;
                let ptr = self.big_a[k][block];
                unsafe { (ptr.add(slot)).as_ref() }
            }
        }
    }

    /// Returns a mutable reference to an element.
    ///
    /// # Time complexity
    ///
    /// O(r).
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        todo!()
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
        todo!()
    }

    /// Returns an iterator over the array.
    ///
    /// The iterator yields all items from start to end.
    pub fn iter(&self) -> OptArrayIter<'_, T> {
        todo!()
    }

    /// Clears the extensible array, removing and dropping all values and
    /// deallocating all previously allocated blocks.
    ///
    /// # Time complexity
    ///
    /// O(n) if elements are droppable, otherwise O(sqrt(n))
    pub fn clear(&mut self) {
        todo!()
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

impl<T> Index<usize> for OptimalArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let Some(item) = self.get(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
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

/// Basic circular buffer, or what Tarjan and Zwick call a cyclic array.
pub struct CyclicArray<T> {
    /// allocated buffer of size `capacity`
    buffer: *mut T,
    /// number of slots allocated in the buffer
    capacity: usize,
    /// offset of the first entry
    head: usize,
    /// number of elements
    count: usize,
}

impl<T> CyclicArray<T> {
    /// Construct a new cyclic array with the given capacity.
    pub fn new(capacity: usize) -> Self {
        let buffer = if capacity == 0 {
            std::ptr::null_mut::<T>()
        } else {
            let layout = Layout::array::<T>(capacity).expect("unexpected overflow");
            unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            }
        };
        Self {
            buffer,
            capacity,
            head: 0,
            count: 0,
        }
    }

    /// Appends an element to the back of the cyclic array.
    ///
    /// # Panic
    ///
    /// Panics if the buffer is already full.
    pub fn push_back(&mut self, value: T) {
        if self.count == self.capacity {
            panic!("cyclic array is full")
        }
        let off = self.physical_add(self.count);
        unsafe { std::ptr::write(self.buffer.add(off), value) }
        self.count += 1;
    }

    /// Prepends an element to the front of the cyclic array.
    ///
    /// # Panic
    ///
    /// Panics if the buffer is already full.
    pub fn push_front(&mut self, value: T) {
        if self.count == self.capacity {
            panic!("cyclic array is full")
        }
        self.head = self.physical_sub(1);
        unsafe { std::ptr::write(self.buffer.add(self.head), value) }
        self.count += 1;
    }

    /// Removes the last element and returns it, or `None` if the cyclic array
    /// is empty.
    pub fn pop_back(&mut self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            self.count -= 1;
            let off = self.physical_add(self.count);
            unsafe { Some(std::ptr::read(self.buffer.add(off))) }
        }
    }

    /// Removes the first element and returns it, or `None` if the cyclic array
    /// is empty.
    pub fn pop_front(&mut self) -> Option<T> {
        if self.count == 0 {
            None
        } else {
            let old_head = self.head;
            self.head = self.physical_add(1);
            self.count -= 1;
            unsafe { Some(std::ptr::read(self.buffer.add(old_head))) }
        }
    }

    /// Provides a reference to the element at the given index.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index < self.count {
            let idx = self.physical_add(index);
            unsafe { Some(&*self.buffer.add(idx)) }
        } else {
            None
        }
    }

    /// Return the number of elements in the array.
    pub fn len(&self) -> usize {
        self.count
    }

    /// Returns the total number of elements the cyclic array can hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Returns true if the array has a length of 0.
    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Returns true if the array has a length equal to its capacity.
    pub fn is_full(&self) -> bool {
        self.count == self.capacity
    }

    /// Perform a wrapping addition relative to the head of the array and
    /// convert the logical offset to the physical offset within the array.
    fn physical_add(&self, addend: usize) -> usize {
        let logical_index = self.head.wrapping_add(addend);
        if logical_index >= self.capacity {
            logical_index - self.capacity
        } else {
            logical_index
        }
    }

    /// Perform a wrapping subtraction relative to the head of the array and
    /// convert the logical offset to the physical offset within the array.
    fn physical_sub(&self, subtrahend: usize) -> usize {
        let logical_index = self
            .head
            .wrapping_sub(subtrahend)
            .wrapping_add(self.capacity);
        if logical_index >= self.capacity {
            logical_index - self.capacity
        } else {
            logical_index
        }
    }
}

impl<T> Index<usize> for CyclicArray<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let Some(item) = self.get(index) else {
            panic!("index out of bounds: {}", index);
        };
        item
    }
}

impl<T> Drop for CyclicArray<T> {
    fn drop(&mut self) {
        use std::ptr::{drop_in_place, slice_from_raw_parts_mut};

        if self.count > 0 && std::mem::needs_drop::<T>() {
            let first_slot = self.physical_add(0);
            let last_slot = self.physical_add(self.count);
            if first_slot < last_slot {
                // elements are in one contiguous block
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(
                        self.buffer.add(first_slot),
                        last_slot - first_slot,
                    ));
                }
            } else {
                // elements wrap around the end of the buffer
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(
                        self.buffer.add(first_slot),
                        self.capacity - first_slot,
                    ));
                    // check if first and last are at the start of the array
                    if first_slot != last_slot || first_slot != 0 {
                        drop_in_place(slice_from_raw_parts_mut(self.buffer, last_slot));
                    }
                }
            }
        }
        // apparently this has no effect if capacity is zero
        let layout = Layout::array::<T>(self.capacity).expect("unexpected overflow");
        unsafe {
            dealloc(self.buffer as *mut u8, layout);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_len_is_empty_when_empty() {
        let sut: OptimalArray<usize> = OptimalArray::new();
        assert_eq!(sut.len(), 0);
        assert!(sut.is_empty());
    }

    #[test]
    fn test_optimal_array_push_many_ints_r3() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..1_000_000 {
            sut.push(value);
        }
        // Rebuild(2B) will have been called 5 times
        assert_eq!(sut.len(), 1_000_000);
        for index in 0..1_000_000 {
            assert_eq!(sut[index], index);
        }
        assert_eq!(sut[99_999], 99_999);
    }

    #[test]
    fn test_optimal_array_push_many_ints_r4() {
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
    }

    #[test]
    fn test_cyclic_array_zero_capacity() {
        let sut = CyclicArray::<usize>::new(0);
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);
        assert!(sut.is_empty());
        assert!(sut.is_full());
    }

    #[test]
    #[should_panic(expected = "cyclic array is full")]
    fn test_cyclic_array_zero_push_panics() {
        let mut sut = CyclicArray::<usize>::new(0);
        sut.push_back(101);
    }

    #[test]
    fn test_cyclic_array_forward() {
        let mut sut = CyclicArray::<usize>::new(10);
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 10);
        assert!(sut.is_empty());
        assert!(!sut.is_full());

        // add until full
        for value in 0..sut.capacity() {
            sut.push_back(value);
        }
        assert_eq!(sut.len(), 10);
        assert_eq!(sut.capacity(), 10);
        assert!(!sut.is_empty());
        assert!(sut.is_full());

        assert_eq!(sut.get(1), Some(&1));
        assert_eq!(sut[1], 1);
        assert_eq!(sut.get(3), Some(&3));
        assert_eq!(sut[3], 3);
        assert_eq!(sut.get(6), Some(&6));
        assert_eq!(sut[6], 6);
        assert_eq!(sut.get(9), Some(&9));
        assert_eq!(sut[9], 9);
        assert_eq!(sut.get(10), None);

        // remove until empty
        for index in 0..10 {
            let maybe = sut.pop_front();
            assert!(maybe.is_some());
            let value = maybe.unwrap();
            assert_eq!(value, index);
        }
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 10);
        assert!(sut.is_empty());
        assert!(!sut.is_full());
    }

    #[test]
    fn test_cyclic_array_backward() {
        let mut sut = CyclicArray::<usize>::new(10);
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 10);
        assert!(sut.is_empty());
        assert!(!sut.is_full());

        // add until full
        for value in 0..sut.capacity() {
            sut.push_front(value);
        }
        assert_eq!(sut.len(), 10);
        assert_eq!(sut.capacity(), 10);
        assert!(!sut.is_empty());
        assert!(sut.is_full());

        // everything is backwards
        assert_eq!(sut.get(1), Some(&8));
        assert_eq!(sut[1], 8);
        assert_eq!(sut.get(3), Some(&6));
        assert_eq!(sut[3], 6);
        assert_eq!(sut.get(6), Some(&3));
        assert_eq!(sut[6], 3);
        assert_eq!(sut.get(9), Some(&0));
        assert_eq!(sut[9], 0);
        assert_eq!(sut.get(10), None);

        // remove until empty
        for index in 0..10 {
            let maybe = sut.pop_back();
            assert!(maybe.is_some());
            let value = maybe.unwrap();
            assert_eq!(value, index);
        }
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 10);
        assert!(sut.is_empty());
        assert!(!sut.is_full());
    }

    #[test]
    #[should_panic(expected = "index out of bounds:")]
    fn test_cyclic_array_index_out_of_bounds() {
        let mut sut = CyclicArray::<usize>::new(10);
        sut.push_back(10);
        sut.push_back(20);
        let _ = sut[2];
    }

    #[test]
    fn test_cyclic_array_drop_partial() {
        let mut sut = CyclicArray::<String>::new(10);
        for _ in 0..7 {
            let value = ulid::Ulid::new().to_string();
            sut.push_back(value);
        }
        drop(sut);
    }

    #[test]
    fn test_cyclic_array_drop_full() {
        let mut sut = CyclicArray::<String>::new(10);
        for _ in 0..sut.capacity() {
            let value = ulid::Ulid::new().to_string();
            sut.push_back(value);
        }
        drop(sut);
    }

    #[test]
    fn test_cyclic_array_drop_wrapped() {
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

    #[test]
    #[should_panic(expected = "cyclic array is full")]
    fn test_cyclic_array_full_panic() {
        let mut sut = CyclicArray::<usize>::new(1);
        sut.push_back(10);
        sut.push_back(20);
    }

    #[test]
    fn test_cyclic_array_wrapping() {
        let mut sut = CyclicArray::<usize>::new(10);
        // push enough to almost fill the buffer
        for value in 0..7 {
            sut.push_back(value);
        }
        // empty the buffer
        while !sut.is_empty() {
            sut.pop_front();
        }
        // push enough to wrap around to the start of the physical buffer
        for value in 0..7 {
            sut.push_back(value);
        }

        assert_eq!(sut.get(1), Some(&1));
        assert_eq!(sut[1], 1);
        assert_eq!(sut.get(3), Some(&3));
        assert_eq!(sut[3], 3);
        assert_eq!(sut.get(6), Some(&6));
        assert_eq!(sut[6], 6);
        assert_eq!(sut.get(8), None);

        // ensure values are removed correctly
        for value in 0..7 {
            assert_eq!(sut.pop_front(), Some(value));
        }
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 10);
        assert!(sut.is_empty());
        assert!(!sut.is_full());
    }
}
