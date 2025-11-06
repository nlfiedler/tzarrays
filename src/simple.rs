//
// Copyright (c) 2025 Nathan Fiedler
//

//! The streamlined implementation of the resizable array which behaves as if
//! the `r` value is fixed at 3.
//!
//! This data structure does not have the additional level of indirection found
//! in the generalized solution, and many of the operations have a constant time
//! complexity since there are only two data block sizes, `B` (small) and `B^2`
//! (large). As such, the performance of this implementation is significantly
//! better than the general version.
//!
//! # Memory Usage
//!
//! An empty resizable array is approximately 128 bytes in size, and while
//! holding elements it will have a space overhead on the order of O(N^1/3) as
//! described in Section 5 of the paper. As elements are added the array will
//! grow by allocating additional data blocks. Likewise, as elements are removed
//! from the end of the array, data blocks will be deallocated as they become
//! empty. At most one empty data block will be retained as an optimization.
//!
//! # Performance
//!
//! The performance and memory usage of this data structure is complicated,
//! please refer to the original paper for details.
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
use std::ptr::{drop_in_place, slice_from_raw_parts_mut};

/// Indicates the size of a data block, either small or large.
#[derive(PartialEq)]
enum BlockSize {
    Small,
    Large,
}

impl fmt::Display for BlockSize {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            BlockSize::Small => write!(f, "sm"),
            BlockSize::Large => write!(f, "lg"),
        }
    }
}

/// A simplified version of the optimal resizable array for which the unused
/// space is on the order of O(N^1/3) plus additional overhead for the data
/// block indices.
///
/// Supports push and pop (and swap/remove) operations only -- insert or remove
/// at other locations is not supported.
///
/// See section 5 of the paper for the details. In short, this is the general
/// solution with a fixed `r` value of 3 and correspondingly streamlined code.
pub struct OptimalArray<T> {
    /// N in the paper (number of elements)
    big_n: usize,
    /// b in the paper, the power of two that yields B, the smallest block size
    /// in terms of elements (see page 13 of the paper)
    little_b: usize,
    /// when N increases to upper_limit, a Rebuild(2B) is required
    upper_limit: usize,
    /// when N decreases to lower_limit, a Rebuild(B/2) is required
    lower_limit: usize,
    /// number of elements in the partially filled block of size B
    n0: usize,
    /// number of small data blocks of size B that are not empty
    n1: usize,
    /// number of large data blocks of size B^2
    n2: usize,
    /// the large blocks of size B^2
    large: CyclicArray<*mut T>,
    /// the small blocks of size B
    small: CyclicArray<*mut T>,
    /// number of empty B blocks, either 0 or 1; avoids deallocating a block and
    /// then allocating another one of the same size for a pop followed by push
    empty: usize,
}

impl<T> OptimalArray<T> {
    /// Return an empty array with zero capacity.
    pub fn new() -> Self {
        // start with B equal to 4 (little b = 2) because maybe the paper
        // suggests that on page 12 and 13; pointers are at least that large so
        // it would be absurd for blocks to be smaller than their pointers
        //
        // always B blocks of size B^2
        let large = CyclicArray::<*mut T>::new(4);
        // always 2B blocks of size B
        let small = CyclicArray::<*mut T>::new(8);
        Self {
            big_n: 0,
            little_b: 2,
            // upper limit is B^r and default B is 4; B^r can be represented as
            // 2^(4r/2), and since r is always 3 the result is simply 64
            upper_limit: 64,
            // set lower_limit to 0 to prevent rebuilding below B=4
            lower_limit: 0,
            n0: 0,
            n1: 0,
            n2: 0,
            large,
            small,
            empty: 0,
        }
    }

    /// Rebuild the indices and blocks with a new (little) b value.
    ///
    /// # Time complexity
    ///
    /// O(n)
    fn rebuild(&mut self, new_b: usize) {
        // prepare the raw parts for building the new array
        let one_b: usize = 1 << new_b;
        let mut n1: usize = 0;
        let mut n2: usize = 0;
        let mut small: CyclicArray<*mut T> = CyclicArray::<*mut T>::new(2 * one_b);
        let mut large: CyclicArray<*mut T> = CyclicArray::<*mut T>::new(one_b);

        // coordinates into the old array that will be advanced as elements are
        // copied into the new array parts, starting with the largest blocks
        // that contain elements
        let mut old_level = if self.n2 > 0 {
            BlockSize::Large
        } else {
            BlockSize::Small
        };
        let old_b: usize = 1 << self.little_b;
        let mut old_block: usize = 0;
        let mut old_slot: usize = 0;
        let mut remaining = self.big_n;

        // copy from the old array into the new raw parts, starting with the
        // largest block size that can be filled with existing data, and moving
        // to the small blocks as less data remains; the assumption is that
        // there will always be some multiple of B elements to be moved and thus
        // everything will always fit snugly into the new blocks
        for new_level in [BlockSize::Large, BlockSize::Small].iter() {
            let (new_index, new_block_len) = match new_level {
                BlockSize::Small => (&mut small, one_b),
                BlockSize::Large => (&mut large, one_b * one_b),
            };
            while remaining >= new_block_len {
                let new_block = if *new_level == BlockSize::Large {
                    n2
                } else {
                    n1
                };
                let layout = Layout::array::<T>(new_block_len).expect("unexpected overflow");
                unsafe {
                    let ptr = alloc(layout).cast::<T>();
                    if ptr.is_null() {
                        handle_alloc_error(layout);
                    }
                    new_index.push_back(ptr);
                }

                // iterate as long as the new block still has capacity
                let mut new_slot: usize = 0;
                while new_slot < new_block_len {
                    let (old_index, old_block_len, old_block_count) = match old_level {
                        BlockSize::Small => (&self.small, old_b, self.n1),
                        BlockSize::Large => (&self.large, old_b * old_b, self.n2),
                    };
                    let copy_len = if old_block_len > new_block_len {
                        new_block_len
                    } else {
                        old_block_len
                    };

                    // copy from old block to new block
                    unsafe {
                        let src = old_index[old_block].add(old_slot);
                        let dst = new_index[new_block].add(new_slot);
                        std::ptr::copy(src, dst, copy_len);
                    }
                    remaining -= copy_len;
                    new_slot += copy_len;
                    old_slot += copy_len;

                    // the old block has been exhausted, move to the next one
                    if old_slot >= old_block_len {
                        // deallocate old block
                        unsafe {
                            let ptr = old_index[old_block];
                            let layout =
                                Layout::array::<T>(old_block_len).expect("unexpected overflow");
                            dealloc(ptr as *mut u8, layout);
                        }
                        old_block += 1;
                        old_slot = 0;
                        if old_block >= old_block_count {
                            // no more old blocks at this level, move on down
                            if old_level == BlockSize::Large {
                                old_level = BlockSize::Small;
                            }
                            // else we will already be done copying anyway
                            old_block = 0;
                        }
                    }
                }

                // account for the newly filled data block
                match new_level {
                    BlockSize::Small => n1 += 1,
                    BlockSize::Large => n2 += 1,
                }
            }
        }

        if self.empty > 0 {
            // an empty block remains at the end of level 1
            let ptr = self.small.pop_back().unwrap();
            let layout = Layout::array::<T>(old_b).expect("unexpected overflow");
            unsafe { dealloc(ptr as *mut u8, layout) }
        }

        // transition to the new array layout
        self.little_b = new_b;
        // the simple array always has an r value of 3
        let r = 3;
        let br = new_b * r;
        // ensure lower_limit is set to prevent rebuilding to B=2
        self.lower_limit = if new_b > 2 { 1 << (br - 2 * r) } else { 0 };
        self.upper_limit = 1 << br;
        // if any B blocks exist, then n[0] must be B because rebuild is called
        // based on the lower/upper bounds which are always multiples of B
        self.n0 = if n1 > 0 { one_b } else { 0 };
        self.n1 = n1;
        self.n2 = n2;
        self.large = large;
        self.small = small;
        self.empty = 0;
    }

    /// Combine small blocks into larger blocks.
    fn combine(&mut self) {
        let one_b: usize = 1 << self.little_b;

        // allocate a new large block
        let new_block_len = one_b * one_b;
        let layout = Layout::array::<T>(new_block_len).expect("unexpected overflow");
        let new_block_ptr = unsafe {
            let ptr = alloc(layout).cast::<T>();
            if ptr.is_null() {
                handle_alloc_error(layout);
            }
            self.large.push_back(ptr);
            ptr
        };

        // copy B small blocks into the new large block
        for j in 0..one_b {
            let src = self.small.pop_front().expect("programming error");
            let dest_slot = j * one_b;
            unsafe {
                let dst = new_block_ptr.add(dest_slot);
                std::ptr::copy(src, dst, one_b);
                let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                dealloc(src as *mut u8, layout);
            }
        }
        self.n1 = one_b;
        self.n2 += 1;
    }

    /// Split large blocks into smaller blocks.
    ///
    /// Called when there are no B sized blocks in the array.
    fn split(&mut self) {
        let one_b: usize = 1 << self.little_b;

        // take one large block and split it into B small blocks
        self.n2 -= 1;
        let old_block_ptr = self.large.pop_back().expect("programming error");
        for j in 0..one_b {
            let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
            let new_block_ptr = unsafe {
                let ptr = alloc(layout).cast::<T>();
                if ptr.is_null() {
                    handle_alloc_error(layout);
                }
                ptr
            };
            self.small.push_back(new_block_ptr);
            self.n1 += 1;
            unsafe {
                let src = old_block_ptr.add(j * one_b);
                std::ptr::copy(src, new_block_ptr, one_b);
            }
        }

        // deallocate the old large block
        let old_block_len = one_b * one_b;
        let layout = Layout::array::<T>(old_block_len).expect("unexpected overflow");
        unsafe { dealloc(old_block_ptr as *mut u8, layout) }
        self.n0 = one_b;
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
        if self.n1 == 2 * one_b && self.n0 == one_b {
            self.combine();
            // combine does nothing to the n0 block, need to fall through to the
            // next condition to ensure we allocate another B block
        }
        if self.n1 == 0 || self.n0 == one_b {
            if self.empty == 0 {
                // A[1][n1] ← Allocate(B)
                let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                unsafe {
                    let ptr = alloc(layout).cast::<T>();
                    if ptr.is_null() {
                        handle_alloc_error(layout);
                    }
                    self.small.push_back(ptr);
                }
                // n1 ← n1 + 1
                self.n1 += 1;
            } else {
                self.empty = 0;
            }
            // n0 ← 0
            self.n0 = 0;
        }
        // A[1][n1−1][n0] ← a
        let ptr = self.small[self.n1 - 1];
        unsafe {
            std::ptr::write(ptr.add(self.n0), value);
        }
        // n0 ← n0 + 1
        self.n0 += 1;
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
        if self.n1 == 0 {
            self.split();
        }

        // n0 ← n0 - 1
        self.n0 -= 1;
        // N ← N - 1
        self.big_n -= 1;

        if self.n0 == 0 {
            let one_b: usize = 1 << self.little_b;
            // if there is another empty data block, deallocate it
            if self.empty == 1 {
                // Deallocate(A[1][n1-1])
                let ptr = self.small.pop_back().unwrap();
                let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                unsafe { dealloc(ptr as *mut u8, layout) }
                // n1 ← n1 - 1
                self.n1 -= 1;
            }
            // leave this last empty data block in case more pushes occur and we
            // would soon be allocating the same sized block again
            self.empty = 1;
            // n0 ← B -- another mistake in the paper?
            if self.n1 > 0 {
                self.n0 = one_b;
            }
        }
    }

    /// Removes the last element from the array and returns it, or `None` if the
    /// array is empty.
    pub fn pop(&mut self) -> Option<T> {
        if self.big_n > 0 {
            // need to copy the value first since shrink() will rearrange the
            // array and possibly deallocate the block containing the element
            let (level, block, slot) = self.locate(self.big_n - 1);
            let ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            };
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
    /// Constant time.
    pub fn capacity(&self) -> usize {
        let one_b: usize = 1 << self.little_b;
        if self.n1 > 0 || self.n2 > 0 {
            one_b * one_b * self.n2 + one_b * self.n1
        } else if self.empty > 0 {
            one_b * self.empty
        } else {
            0
        }
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
    /// Constant time.
    fn locate(&self, index: usize) -> (BlockSize, usize, usize) {
        let small_size: usize = 1 << self.little_b;
        let large_size: usize = small_size * small_size;
        let last_large = large_size * self.n2;
        if index < last_large {
            // among the large blocks
            let block = index / large_size;
            let slot = index % large_size;
            (BlockSize::Large, block, slot)
        } else {
            // among the small blocks
            let s_index = index - last_large;
            let block = s_index / small_size;
            let slot = s_index % small_size;
            (BlockSize::Small, block, slot)
        }
    }

    /// Retrieve a reference to the element at the given offset.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.big_n {
            None
        } else {
            let (level, block, slot) = self.locate(index);
            let ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            };
            unsafe { (ptr.add(slot)).as_ref() }
        }
    }

    /// Returns a mutable reference to an element.
    ///
    /// # Time complexity
    ///
    /// Constant time.
    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        if index >= self.big_n {
            None
        } else {
            let (level, block, slot) = self.locate(index);
            let ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            };
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
    /// Constant time.
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
            let index_ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            }
            .add(slot);
            let value = index_ptr.read();
            // find the pointer of the last element and copy to index pointer
            let (level, block, slot) = self.locate(self.big_n - 1);
            let last_ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            }
            .add(slot);
            std::ptr::copy(last_ptr, index_ptr, 1);
            self.shrink();
            value
        }
    }

    // Returns an iterator over the array.
    //
    // The iterator yields all items from start to end.
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
    /// O(n) if elements are droppable, otherwise O(N^(1/3))
    pub fn clear(&mut self) {
        let one_b: usize = 1 << self.little_b;

        if self.big_n > 0 && std::mem::needs_drop::<T>() {
            // drop items and deallocate the data blocks

            // smallest block needs special care
            if self.n0 > 0 {
                let ptr = self.small.pop_back().unwrap();
                unsafe {
                    drop_in_place(slice_from_raw_parts_mut(ptr, self.n0));
                    let layout = Layout::array::<T>(one_b).expect("unexpected overflow");
                    dealloc(ptr as *mut u8, layout);
                }
            }

            // drop all elements in all remaining blocks
            for level in [BlockSize::Large, BlockSize::Small].iter() {
                let (block_index, block_len) = match level {
                    BlockSize::Small => (&mut self.small, one_b),
                    BlockSize::Large => (&mut self.large, one_b * one_b),
                };
                while let Some(ptr) = block_index.pop_front() {
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(ptr, block_len));
                        let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            }
        } else {
            // If elements do not need dropping, then simply deallocate the data
            // blocks. Note that even an "empty" array may still have an empty
            // data block lingering that needs to be freed.
            for level in [BlockSize::Large, BlockSize::Small].iter() {
                let (block_index, block_len) = match level {
                    BlockSize::Small => (&mut self.small, one_b),
                    BlockSize::Large => (&mut self.large, one_b * one_b),
                };
                while let Some(ptr) = block_index.pop_front() {
                    unsafe {
                        let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
                        dealloc(ptr as *mut u8, layout);
                    }
                }
            }
        }

        // zero out everything to the initial state
        self.big_n = 0;
        self.empty = 0;
        self.little_b = 2;
        self.upper_limit = 64;
        self.lower_limit = 0;
        self.n0 = 0;
        self.n1 = 0;
        self.n2 = 0;
    }
}

impl<T> Default for OptimalArray<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> fmt::Display for OptimalArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OptimalArray(N: {}, l: {}, h: {}, b: {}, e: {}, n0: {}, n1: {}, n2: {})",
            self.big_n,
            self.lower_limit,
            self.upper_limit,
            self.little_b,
            self.empty,
            self.n0,
            self.n1,
            self.n2
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
        let small = std::mem::take(&mut me.small);
        let large = std::mem::take(&mut me.large);
        let small_size: usize = 1 << me.little_b;
        let large_size: usize = small_size * small_size;
        let last_large = large_size * me.n2;
        OptArrayIntoIter {
            cursor: 0,
            big_n: me.big_n,
            small_size,
            large_size,
            last_large,
            small,
            large,
        }
    }
}

/// An iterator that moves out of an optimal array.
pub struct OptArrayIntoIter<T> {
    /// offset into the array
    cursor: usize,
    /// N in the paper (number of elements)
    big_n: usize,
    /// number of elements in small blocks
    small_size: usize,
    /// number of elements in large blocks
    large_size: usize,
    /// index of the last large block
    last_large: usize,
    /// the large blocks of size B^2
    large: CyclicArray<*mut T>,
    /// the small blocks of size B
    small: CyclicArray<*mut T>,
}

impl<T> OptArrayIntoIter<T> {
    fn locate(&self, index: usize) -> (BlockSize, usize, usize) {
        if index < self.last_large {
            // among the large blocks
            let block = index / self.large_size;
            let slot = index % self.large_size;
            (BlockSize::Large, block, slot)
        } else {
            // among the small blocks
            let s_index = index - self.last_large;
            let block = s_index / self.small_size;
            let slot = s_index % self.small_size;
            (BlockSize::Small, block, slot)
        }
    }
}

impl<T> Iterator for OptArrayIntoIter<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor < self.big_n {
            let (level, block, slot) = self.locate(self.cursor);
            self.cursor += 1;
            let ptr = match level {
                BlockSize::Small => self.small[block],
                BlockSize::Large => self.large[block],
            };
            unsafe { Some((ptr.add(slot)).read()) }
        } else {
            None
        }
    }
}

impl<T> Drop for OptArrayIntoIter<T> {
    fn drop(&mut self) {
        // all visited elements have already been dropped, so carefully examine
        // all of the blocks and determine which ones have unvisited elements
        let (first_level, first_block, first_slot) = self.locate(self.cursor);
        let (last_level, last_block, last_slot) = if self.big_n == 0 {
            (BlockSize::Small, 0, 0)
        } else {
            self.locate(self.big_n - 1)
        };
        let not_finished = self.cursor < self.big_n;
        let levels = [
            (BlockSize::Large, &mut self.large, self.large_size),
            (BlockSize::Small, &mut self.small, self.small_size),
        ];
        for (level, block_index, block_len) in levels.into_iter() {
            let layout = Layout::array::<T>(block_len).expect("unexpected overflow");
            let mut block = 0;
            while let Some(ptr) = block_index.pop_front() {
                let unvisited = (level != first_level || block >= first_block)
                    && (level != last_level || block < last_block);
                let is_first = level == first_level && block == first_block;
                let is_last = level == last_level && block == last_block;
                if is_first && is_last {
                    // special-case for first block being the last block
                    if first_slot <= last_slot {
                        unsafe {
                            drop_in_place(slice_from_raw_parts_mut(
                                ptr.add(first_slot),
                                last_slot - first_slot + 1,
                            ));
                        }
                    }
                } else if is_first {
                    // special-case for partially visited first block
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(
                            ptr.add(first_slot),
                            block_len - first_slot,
                        ));
                    }
                } else if is_last && not_finished {
                    // special-case for maybe partially filled last block
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(ptr, last_slot + 1));
                    }
                } else if unvisited {
                    // all other unvisited blocks are dropped in their entirety
                    unsafe {
                        drop_in_place(slice_from_raw_parts_mut(ptr, block_len));
                    }
                }
                unsafe {
                    dealloc(ptr as *mut u8, layout);
                }
                block += 1;
            }
        }

        // zero out everything to the initial state
        self.cursor = 0;
        self.big_n = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_array_when_empty() {
        let sut: OptimalArray<usize> = OptimalArray::new();
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 0);
        assert!(sut.is_empty());
    }

    #[test]
    fn test_simple_array_capacity() {
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
    fn test_simple_array_push_within_capacity() {
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
    fn test_simple_array_get_mut_index_mut() {
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
    fn test_simple_array_iter() {
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..1000 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 1000);
        for (index, value) in sut.iter().enumerate() {
            assert_eq!(sut[index], *value);
        }
    }

    #[test]
    fn test_simple_array_push_many_ints() {
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
    fn test_simple_array_grow_shrink_empty_block() {
        // test the empty block reuse logic for shrink and grow
        //
        // default B is 4, so push 12, then pop 8, then push 8 more, then ensure
        // all values are present
        let mut sut: OptimalArray<usize> = OptimalArray::new();
        for value in 0..12 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 12);
        for _ in 0..8 {
            sut.pop();
        }
        assert_eq!(sut.len(), 4);
        for value in 4..12 {
            sut.push(value);
        }
        assert_eq!(sut.len(), 12);
        for (idx, elem) in sut.iter().enumerate() {
            assert_eq!(idx, *elem);
        }

        // try to trigger any clear/drop logic
        sut.clear();
    }

    #[test]
    fn test_simple_array_pop_if() {
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
    fn test_simple_array_swap_remove_single_block() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        assert_eq!(sut.len(), 1);
        let one = sut.swap_remove(0);
        assert_eq!(one, 1);
        assert_eq!(sut.len(), 0);
    }

    #[test]
    fn test_simple_array_swap_remove_multiple_blocks() {
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
    fn test_simple_array_swap_remove_panic_empty() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.swap_remove(0);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 1) should be < len (is 1)")]
    fn test_simple_array_swap_remove_panic_range_edge() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        sut.swap_remove(1);
    }

    #[test]
    #[should_panic(expected = "swap_remove index (is 2) should be < len (is 1)")]
    fn test_simple_array_swap_remove_panic_range_exceed() {
        let mut sut: OptimalArray<u32> = OptimalArray::new();
        sut.push(1);
        sut.swap_remove(2);
    }

    #[test]
    fn test_simple_array_from_iterator() {
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
    fn test_simple_array_clear_and_reuse_tiny() {
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
    fn test_simple_array_clear_and_reuse_ints() {
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
    fn test_simple_array_clear_and_reuse_strings() {
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
    fn test_simple_array_push_many_strings() {
        let mut sut = OptimalArray::<String>::new();
        for _ in 0..1_000_000 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        assert_eq!(sut.len(), 1_000_000);
        assert_eq!(sut.capacity(), 1_000_064);
        while let Some(value) = sut.pop() {
            assert_eq!(value.len(), 26);
        }
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 4);

        // and do it again to be sure shrinking works correctly
        for _ in 0..1_000_000 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        assert_eq!(sut.len(), 1_000_000);
        assert_eq!(sut.capacity(), 1_000_064);
        while let Some(value) = sut.pop() {
            assert_eq!(value.len(), 26);
        }
        assert_eq!(sut.len(), 0);
        assert_eq!(sut.capacity(), 4);
    }

    #[test]
    fn test_simple_array_push_get_many_instances_ints() {
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
    fn test_simple_array_push_get_many_instances_strings() {
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
    fn test_simple_array_into_iterator_drop_empty() {
        let sut: OptimalArray<String> = OptimalArray::new();
        assert_eq!(sut.into_iter().count(), 0);
    }

    #[test]
    fn test_simple_array_into_iterator_edge_case() {
        // add 4, iterate 3
        let inputs = ["one", "two", "three", "four"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..3 {
            iter.next();
        }

        // add 4, iterate 4
        let inputs = ["one", "two", "three", "four"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        assert_eq!(4, sut.into_iter().count());

        // add 8, iterate 3
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..3 {
            iter.next();
        }

        // add 8, iterate 5
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..5 {
            iter.next();
        }

        // add 8, iterate 8
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        assert_eq!(8, sut.into_iter().count());

        // add 7, iterate 3
        let inputs = ["one", "two", "three", "four", "five", "six", "seven"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..3 {
            iter.next();
        }

        // add 7, iterate 5
        let inputs = ["one", "two", "three", "four", "five", "six", "seven"];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..5 {
            iter.next();
        }

        // add 13, iterate 3
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..3 {
            iter.next();
        }

        // add 13, iterate 5
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..5 {
            iter.next();
        }

        // add 13, iterate 9
        let inputs = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen",
        ];
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for item in inputs {
            sut.push(item.to_owned());
        }
        let mut iter = sut.into_iter();
        for _ in 0..9 {
            iter.next();
        }
    }

    #[test]
    fn test_simple_array_into_iterator_ints_done() {
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
    fn test_simple_array_into_iterator_drop_tiny_done() {
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
    fn test_simple_array_into_iterator_drop_tiny_partial() {
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
    fn test_simple_array_into_iterator_drop_all() {
        let mut sut: OptimalArray<String> = OptimalArray::new();
        for _ in 0..512 {
            let value = ulid::Ulid::new().to_string();
            sut.push(value);
        }
        let _ = sut.into_iter();
    }

    #[test]
    fn test_simple_array_into_iterator_drop_large_partial() {
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
    fn test_simple_array_into_iterator_drop_large_more_partial() {
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
