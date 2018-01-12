pub mod dyn_bit_vec {

extern crate bit_field;
extern crate rayon;
use std::vec::Vec;
use std::fmt;
use self::bit_field::BitField;
use self::rayon::prelude::*;
use std::u32::MAX;

#[derive(Eq, PartialEq)]
pub struct DBVec {
	words: Vec<u32>,
	len_rem: u8,	// length = (words.length - 1) * 32 + len_rem
}

impl DBVec {
	pub fn new() -> Self {
		DBVec {
			words: Vec::new(),
			len_rem: 0,
		}
	}

	pub fn from_u32_slice(slice: &[u32]) -> Self {
		let mut temp_vec = Vec::with_capacity(slice.len());
		temp_vec.extend_from_slice(slice);
		DBVec {
			words: temp_vec,
			len_rem: 0,
		}
	}

	pub fn from_bytes(bytes: &[u8]) -> Self {
		let mut temp_vec: Vec<u32> = Vec::with_capacity(bytes.len() * 4);
		let mut temp_int = 0u32;
		for byte_pos in bytes.iter().enumerate() {
			temp_int = temp_int | (*byte_pos.1 as u32);
			if byte_pos.0 % 4 == 3 {
				temp_vec.push(temp_int);
				temp_int = 0;
			}
			temp_int = temp_int << 8;
			println!("{:032b}", temp_int); 
		}
		if bytes.len() % 4 != 0 {
			temp_int = temp_int >> 8;
			temp_vec.push(temp_int);
		}
		DBVec {
			words: temp_vec,
			len_rem: ((bytes.len() * 8) % 32) as u8,
		}
	}

	pub fn len(&self) -> u64 {
		// FIXME: error when initialised from slice
		match self.words.len() {
			0 => self.len_rem as u64,
			_ => {
					if self.len_rem == 0 {
						(self.words.len() * 32) as u64
					} else {
						((self.words.len() - 1) * 32) as u64 + self.len_rem as u64
					}
				 }
		}
	}

	fn inc_len(&mut self) {
		self.len_rem += 1;
		if self.len_rem == 32 || self.words.len() == 0 {
			self.words.push(0);
			if self.len_rem == 32 {
				self.len_rem = 0;
			}
		}
	}

	pub fn pop_cnt(&self) -> u64 {
		if self.words.len() < 1000000 {
			self.pop_cnt_words()
		} else {
			self.pop_cnt_words_parallel()
		}
	}

	pub fn pop_cnt_words(&self) -> u64 {
		self.words.iter().fold(0, |nr_bits, word| nr_bits + word.count_ones() as u64)
	}

	pub fn pop_cnt_words_parallel(&self) -> u64 {
		self.words.par_iter()
			.map(|word| word.count_ones() as u64)
			.sum()
	}

	// insert a bit at position 'index'
	pub fn insert(&mut self, bit: bool, index: u64) {
		if index > self.len() {
			panic!("Index out of bound: index = {} while the length is {}", index, self.len());
		}
		self.inc_len();
		let bit_index = (index % 32) as usize;
		let word_index = (index / 32) as usize;

		let mut last_bit = false;
		// change the word that has to be changed
		if let Some(word) = self.words.get_mut(word_index) {
			last_bit = word.get_bit(31);
			Self::insert_in_word(word, bit_index, bit);
		} 

		// for every word from word_index + 1 until end: shift left; put last_bit as first bit; remember last_bit etc
		let word_iter = self.words.iter_mut().skip(word_index + 1);
		for word in word_iter {
			let first_bit = last_bit;
			last_bit = word.get_bit(31);
			*word = *word << 1;
			*word.set_bit(0, first_bit);
		}
	}

	// push a bit to the end. This can slightly more efficient than insert(bit, len())
	// because insert requires additional checks
	pub fn push(&mut self, bit: bool) {
		let bit_index = (self.len() % 32) as usize;
		self.inc_len();
		if bit {
			let word_index = (self.len() / 32) as usize;
			if let Some(word) = self.words.get_mut(word_index) {
				word.set_bit(bit_index, bit);
			}
		}
	}

	pub fn insert_vec(&mut self, other: &Self, index: u64) {
		if index > self.len() {
			panic!("Index out of bound: index = {} while the length is {}", index, self.len());
		}
		// determine insertion point
		let insertion_word_index = index / 32;
		let start_insertion_bit_index = (index % 32) as u8;
		let end_insertion_bit_index = start_insertion_bit_index + (other.len() % 32) as u8;

		// prepare self
		
	}

	// insert a bit in a given word at index bit_index. The bits after bit_index shift one place towards the end
	#[inline]
	fn insert_in_word(word: &mut u32, bit_index: usize, bit: bool) {
		for remaining_bit_index in (bit_index + 1..32).rev() {
			let prev_bit = word.get_bit(remaining_bit_index - 1);
			word.set_bit(remaining_bit_index, prev_bit);
		}
		word.set_bit(bit_index, bit);
	}

	// Returns true if 'other' is a subvector of 'self', starting at index 0.
	// Equal vectors start with each other.
	pub fn starts_with(&self, other: &Self) -> bool {
		if other.len() > self.len() {
			false
		} else if other.len() == 0 {
			true
		} else {
			let common_word_len = other.words.len() - 1;
			// may be parallellized for large vectors?
			let self_words = &self.words[..common_word_len];
			let other_words = &other.words[..common_word_len];
			if self_words == other_words {
				let self_last_word = self.words.last().unwrap();
				let other_last_word = other.words.get(common_word_len).unwrap();
				let bits_to_compare = (other.len() % 32) as usize;
				if bits_to_compare > 0 {
					let self_bits = self_last_word.get_bits(0..bits_to_compare);
					let other_bits = other_last_word.get_bits(0..bits_to_compare);
					self_bits == other_bits
				} else {
					true
				}
			} else {
				false
			}
		}
	}

	// Shifts everything nr_bits (max 31 bits) towards the end of the vector.
	// This means nr_bits leading zero's are introduced; the vector grows.
	// Overflowing bits are put into a new word at the end of the vector.
	pub fn align_to_end(&mut self, nr_bits: u8) {
		if self.len_rem == 0 {
			self.words.push(0u32);
		}
		let overflowing_bits = (MAX >> nr_bits) ^ MAX;

		// check if next word needed? self.len_rem + nr_bits > 32 ???
		self.len_rem += nr_bits;
		if self.len_rem > 32 {
			self.words.push(0u32);
			self.len_rem = self.len_rem % 32;
		}

		// now do the trick. rotate each word to left, put the 'overflow' into the next word
		let mut overflow = 0u32;
		for word in self.words.iter_mut() {
			let new_overflow = (*word & overflowing_bits) >> (32 - nr_bits);
			*word = (*word << nr_bits) | overflow;
			overflow = new_overflow;
		}
	}

	// Shifts everything nr_bits (max 31 bits) towards the beginning of the vector; the vector shrinks.
	pub fn shift_to_begin(&mut self, nr_bits: u8) {
		let underflowing_bits = (MAX << nr_bits) ^ MAX;
		let mut underflow = 0u32;
		for word in self.words.iter_mut().rev() {
			let new_underflow = (*word & underflowing_bits) << (32 - nr_bits);
			*word = (*word >> nr_bits) | underflow;
			underflow = new_underflow;
		}

		// check if last word can be deleted
		if self.len_rem == 0 {
			self.len_rem = 32;
		}else if self.len_rem <= nr_bits {
			self.len_rem += 32;
			self.words.pop();
		}
		self.len_rem -= nr_bits;
	}

	// split the vector at index 'at'. DOES NOT ALIGN SECOND PART!!
	pub fn split(&mut self, at: u64) -> Self {
		println!("Input: {:?}", self);
		// just split the words vector
		let at_word = at / 32;
		let mut other_words = self.words.split_off(at_word as usize);
		self.words.shrink_to_fit();
		println!("Other_words: {:?}", other_words);
		println!("Input: {:?}", self);

		// put the first relevant bits of other_words at the end of self.words
		let start_insertion_bit_index = (at % 32) as u8;
		let other_bit_mask = MAX << start_insertion_bit_index;
		println!("Other bitmask:        {:032b}", other_bit_mask);
		let self_bit_mask = other_bit_mask ^ MAX;
		println!("Self bitmask :        {:032b}", self_bit_mask);
		if let Some(first_of_other) = other_words.first_mut() {
			let last_of_self = *first_of_other & self_bit_mask;
			println!("last_of_self:         {:032b}", last_of_self);
			*first_of_other = *first_of_other & other_bit_mask;
			println!("first_of_other:       {:032b}", *first_of_other);
			self.words.push(last_of_self);
		}
		println!("Input: {:?}", self);
		DBVec {
			words: other_words,
			len_rem: 0
		}
	}
}

impl fmt::Debug for DBVec {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "DBVec: ({}, {}, ", self.len(), self.pop_cnt());
		for word in self.words.iter() {
			write!(f, "{:032b} ", word);
		}
		write!(f, ")")
	}
}

}

#[cfg(test)]
mod tests {

use dyn_bit_vec::DBVec;
use std::u16::MAX;

	#[test]
	fn from_u32_slice() {
		let vec = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		println!("{:?}", vec);
		assert_eq!(vec.len(), 96);
		assert_eq!(vec.pop_cnt(), 3);
	}

	#[test]
	fn insert() {
		let mut vec = DBVec::new();
		vec.insert(true, 0);
		println!("{:?}", vec);
		for _ in 1..42 {
			vec.insert(true, 1);
			println!("{:?}", vec);
		}
		vec.insert(false, 42);
		vec.insert(true, 43);
		println!("{:?}", vec);
		for c in 42..62 {
			vec.insert(true, c);
			println!("{:?}", vec);
		}
	}

	#[test]
	fn push() {
		let mut vec = DBVec::new();
		let mut bit = true;
		for _ in 1..42 {
			vec.push(bit);
			bit = !bit;
			println!("{:?}", vec);
		}
	}

	#[test]
	fn from_bytes() {
		let vec = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		//let vec = DBVec::from_bytes(&[0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000]);
		//let vec = DBVec::from_bytes(&[0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000]);
		println!("{:?}", vec);
		assert_eq!(48, vec.len());
		assert_eq!(11, vec.pop_cnt());
	}

	#[test]
	fn test_eq() {
		let mut vec1 = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		let vec2 = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		assert_eq!(vec1, vec2);
		vec1.push(true);
		assert!(vec1 != vec2);
	}

	#[test]
	fn starts_with() {
		// test 1: empty vecs
		let vec1 = DBVec::new();
		let vec2 = DBVec::new();
		assert!(vec1.starts_with(&vec2));
		assert!(vec2.starts_with(&vec1));

		// test 2: same vectors, one word long
		let vec3 = DBVec::from_u32_slice(&[1025]);
		let vec4 = DBVec::from_u32_slice(&[1025]);
		assert!(vec3.starts_with(&vec4));
		assert!(vec4.starts_with(&vec3));

		// test 3: same vectors, length not on word boundary
		let vec5 = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		let vec6 = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		assert!(vec5.starts_with(&vec6));
		assert!(vec6.starts_with(&vec5));

		// test 4: different vectors
		let vec7 = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10100000]);
		let vec8 = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		assert!(vec8.starts_with(&vec7));
		assert!(!vec7.starts_with(&vec8));
	}

	#[test]
	fn align_to_end() {
		let mut vec = DBVec::new();
		vec.push(true);
		vec.push(true);
		println!("{:?}", vec);
		vec.align_to_end(5);
		println!("{:?}", vec);
		vec.align_to_end(20);
		println!("{:?}", vec);
		vec.align_to_end(6);
		println!("{:?}", vec);
		vec.align_to_end(31);
		println!("{:?}", vec);
		vec.align_to_end(1);
		println!("{:?}", vec);

		let mut vec2 = DBVec::from_u32_slice(&[1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]);
		println!("{:?}", vec2);
		vec2.align_to_end(31);
		println!("{:?}", vec2);
	}

	#[test]
	fn shift_to_begin() {
		let mut vec = DBVec::from_u32_slice(&[0b10000000_00000000_00000000_00001000u32]);
		println!("{:?}", vec);
		vec.shift_to_begin(3);
		println!("{:?}", vec);
		vec.shift_to_begin(1);
		println!("{:?}", vec);

		let mut vec2 = DBVec::from_u32_slice(&[1, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000]);
		println!("{:?}", vec2);
		vec2.shift_to_begin(31);
		println!("{:?}", vec2);
		vec2.shift_to_begin(1);
		println!("{:?}", vec2);
	}

	#[test]
	fn align_and_shift() {
		let mut vec = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		let vec_result = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		vec.align_to_end(25);
		vec.shift_to_begin(25);
		assert_eq!(vec, vec_result);
	}

	#[test]
	fn split_off() {
		let mut vec = DBVec::from_u32_slice(&[0b10000000_10000000_00000000_00001000u32]);
		let vec2 = vec.split_off(4);
		println!("{:?}", vec2);
	}
}
