//
// Copyright (c) 2025 Nathan Fiedler
//

//! An implementation of resizable arrays as described in the paper **Optimal
//! resizable arrays** by Tarjan and Zwick, published in 2023.
//!
//! * <https://doi.org/10.48550/arXiv.2211.11009>
//!
//! # General versus Simple
//!
//! There are two implementations of the array described in the paper available
//! in this crate.
//!
//! The `simple` version is as described in Section 5 of the paper, which is
//! little more than a streamlined version of the general implementation with an
//! upper bound on the overhead of O(N^(1/3)). Because the code is simpler and
//! avoids loops, the time complexity of many of the operations is constant.
//!
//! Meanwhile, the `general` implementation allows for different values for `r`
//! (2 or higher) and as such provides a means of reducing the unused space to
//! small fractions of the overall array size. That flexibility comes at a cost
//! to the time complexity, as now nearly all operations involve looping over
//! the variable number of indices of data blocks of differing sizes.
//!
//! There is also an implementation of a simple circular buffer, named
//! `CyclicArray` because that is what the paper calls it. Its capacity is fixed
//! at the time of construction and it will panic if the capacity is exceeded.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error};
use std::ops::Index;

pub mod general;
pub mod simple;

/// Basic circular buffer, or what Tarjan and Zwick call a cyclic array.
///
/// Unlike the `VecDeque` in the standard library, this array has a fixed size
/// and will panic if a push is performed while the array is already full.
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

    /// Clears the cyclic array, removing and dropping all values.
    pub fn clear(&mut self) {
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
        self.head = 0;
        self.count = 0;
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

impl<T> Default for CyclicArray<T> {
    fn default() -> Self {
        Self::new(0)
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
        self.clear();
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
    fn test_cyclic_array_clear_and_reuse() {
        let mut sut = CyclicArray::<String>::new(10);
        for _ in 0..7 {
            let value = ulid::Ulid::new().to_string();
            sut.push_back(value);
        }
        sut.clear();
        for _ in 0..7 {
            let value = ulid::Ulid::new().to_string();
            sut.push_back(value);
        }
        sut.clear();
        for _ in 0..7 {
            let value = ulid::Ulid::new().to_string();
            sut.push_back(value);
        }
        sut.clear();
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
