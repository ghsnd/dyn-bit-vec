extern crate bit_field;
extern crate rayon;
use std::vec::Vec;
use std::fmt;
use self::bit_field::BitField;
use self::rayon::prelude::*;
use std::u32::MAX;

#[derive(Clone, Eq, PartialEq)]
pub struct DBVec {
	words: Vec<u32>,
	len_rem: u8,	// length = (words.length - 1) * 32 + len_rem
}

impl DBVec {

	///// constructors /////

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

	pub fn from_elem(nbits: usize, bit: bool) -> Self {
		let elem = match bit {
			false => 0,
			true  => MAX
		};
		let len = nbits / 32 + 1;
		let rem = nbits % 32;
		let mut word_vec = vec![elem; len];
		if rem > 0 {
			if let Some(last_word) = word_vec.last_mut() {
				*last_word = *last_word >> (32 - rem);
			}
		} else {
			word_vec.pop();
		}
		DBVec {
			words: word_vec,
			len_rem: rem as u8
		}
	}

	////////////////////////

	pub fn words(&self) -> &Vec<u32> {
		&self.words
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

	// get the value of the bit at position 'index'
	pub fn get(&self, index: u64) -> bool {
		if index >= self.len() {
			panic!("Index out of bounds: index = {} while the length is {}", index, self.len());
		}
		let bit_index = (index % 32) as usize;
		let word_index = (index / 32) as usize;
		if let Some(word) = self.words.get(word_index) {
			word.get_bit(bit_index)
		} else {
			panic!("Should not occur!");
		}
	}

	// insert a bit at position 'index'
	pub fn insert(&mut self, bit: bool, index: u64) {
		if index > self.len() {
			panic!("Index out of bounds: index = {} while the length is {}", index, self.len());
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

	pub fn insert_vec(&mut self, other: &mut Self, index: u64) {
		let self_len = self.len();
		if index > self_len {
			panic!("Index out of bound: index = {} while the length is {}", index, self_len);
		}

		let new_len_rem = (self.len_rem + other.len_rem) % 32;

		// determine insertion point
		let start_insertion_bit_index = (index % 32) as u8;
		let end_insertion_bit_index = (start_insertion_bit_index + other.len_rem) % 32;

		let mut self_tail_vec = self.split(index);
		other.align_to_end(start_insertion_bit_index);
		if start_insertion_bit_index < end_insertion_bit_index {
			let shift_amount = end_insertion_bit_index - start_insertion_bit_index;
			self_tail_vec.align_to_end(shift_amount);
		} else if start_insertion_bit_index > end_insertion_bit_index{
			let shift_amount = start_insertion_bit_index - end_insertion_bit_index;
			self_tail_vec.shift_to_begin(shift_amount);
			self_tail_vec.words.pop();
		}

		// 'merge' last word of first part of self with first word of other
		if let Some(last_of_first_part) = self.words.last() {
			if let Some(first_other) = other.words.first_mut() {
				*first_other = *last_of_first_part | *first_other;
			}
		}
		self.words.pop();

		// 'merge' last word of other with first word of last part of self_tail_vec
		if let Some(last_of_other) = other.words.last() {
			if let Some(first_tail) = self_tail_vec.words.first_mut() {
				*first_tail = *last_of_other | *first_tail;
			}
		}
		other.words.pop();

		//merge vectors
		self.words.append(&mut other.words);
		self.words.append(&mut self_tail_vec.words);
		self.len_rem = new_len_rem;
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
		if nr_bits > 0 {
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
	}

	// split the vector at index 'at'. DOES NOT ALIGN SECOND PART!!
	pub fn split(&mut self, at: u64) -> Self {
		// just split the words vector
		let at_word = at / 32;
		let mut other_words = self.words.split_off(at_word as usize);
		self.words.shrink_to_fit();

		// put the first relevant bits of other_words at the end of self.words
		let start_insertion_bit_index = (at % 32) as u8;
		let other_bit_mask = MAX << start_insertion_bit_index;
		let self_bit_mask = other_bit_mask ^ MAX;
		if let Some(first_of_other) = other_words.first_mut() {
			let last_of_self = *first_of_other & self_bit_mask;
			*first_of_other = *first_of_other & other_bit_mask;
			self.words.push(last_of_self);
		}
		DBVec {
			words: other_words,
			len_rem: 0
		}
	}

	// returns the longest common prefix of self and the other
	pub fn longest_common_prefix (&self, other: &DBVec) -> DBVec {
		let mut common_words: Vec<u32> = Vec::new();
		let zipped_iter = self.words.iter().zip(other.words.iter());
		let mut len_rem = 0;
		for word_pair in zipped_iter {
			if word_pair.0 == word_pair.1 {
				common_words.push(*word_pair.0);
			} else {
				let mut result: u32 = 0;
				let mut do_push = false;
				for bit_nr in 0..32 {
					let bit = word_pair.0.get_bit(bit_nr);
					if bit == word_pair.1.get_bit(bit_nr) {
						result.set_bit(bit_nr, bit);
						len_rem = bit_nr;
						do_push = true;
					} else {
						break;
					}
				}
				if do_push {
					common_words.push(result);
				}
				break;
			}
		}
		DBVec {
			words: common_words,
			len_rem: 0
		}
	}

	pub fn different_suffix(&self, at: u64) -> (bool, Self) {
		let first_bit = self.get(at);
		let new_at = at + 1;
		let at_word = (new_at / 32) as usize;
		let at_bit = (new_at % 32) as u8;
		println!("  at_bit: {}", at_bit);
		let mut result_vec = DBVec::from_u32_slice(&self.words[at_word..]);
		result_vec.shift_to_begin(at_bit);
		result_vec.len_rem = ((self.len() - new_at) % 32) as u8;
		(first_bit, result_vec)
	}

}

impl fmt::Debug for DBVec {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "DBVec: ({}, {}, ", self.len(), self.pop_cnt());
		let mut count = 0u8;
		for word in self.words.iter() {
			write!(f, "{:032b} ", word);
			count += 1;
			if count == 100 {
				count = 1;
				write!(f, "\n");
			}
		}
		write!(f, ")")
	}
}

#[cfg(test)]
mod tests {

use DBVec;

	#[test]
	fn from_u32_slice() {
		let vec = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		println!("{:?}", vec);
		assert_eq!(vec.len(), 96);
		assert_eq!(vec.pop_cnt(), 3);
	}

	#[test]
	fn from_elem() {
		let mut vec1 = DBVec::from_elem(35, false);
		println!("{:?}", vec1);
		let mut vec2 = DBVec::from_elem(35, true);
		println!("{:?}", vec2);
		vec2.insert_vec(&mut vec1, 30);
		println!("{:?}", vec2);
		assert_eq!(vec2.len(), 70);
		assert_eq!(vec2.words(), &[0b00111111111111111111111111111111, 0b00000000000000000000000000000000, 0b00000000000000000000000000111110]);

		let vec_test = DBVec::from_elem(32, true);
		assert_eq!(DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]), vec_test);
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
	fn split() {
		let mut vec = DBVec::from_elem(35, true);
		let vec2 = vec.split(30);
		println!("{:?}", vec2);
	}

	#[test]
	fn insert_vec() {
		let mut vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		let mut vec2 = DBVec::from_bytes(&[0b01111110]);
		//println!("vec1: {:?}", vec1);
		//println!("vec2: {:?}", vec2);
		vec1.insert_vec(&mut vec2, 4);
		println!("vec1: {:?}", vec1);
		//println!("vec2: {:?}", vec2);
		assert_eq!(vec1.words(), &[0b11111111111111111111011111101111, 0b00000000000000000000000011111111]);
		let mut vec3 = DBVec::from_u32_slice(&[256, 256, 256, 256, 256, 256, 256, 256, 256]);
		println!("vec3: {:?}", vec3);
		vec1.insert_vec(&mut vec3, 34);
		println!("vec1: {:?}", vec1);
		assert_eq!(vec1.len(), 328);
		assert_eq!(vec1.words(), &[0b11111111111111111111011111101111, 0b00000000000000000000010000000011,
			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
			0b00000000000000000000000011111100]);
	}

	#[test]
	fn longest_common_prefix() {
		// simple one-word vectors
		let vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		let vec2 = DBVec::from_u32_slice(&[0b11111111_11111111_11111011_11111111u32]);
		let exp  = DBVec::from_u32_slice(                        &[0b11_11111111u32]);
		let result = vec1.longest_common_prefix(&vec2);
		assert_eq!(exp, result);

		// schould be long_vec_1
		let long_vec1 = DBVec::from_u32_slice(&[256, 256, 256, 256, 256, 256, 256, 256, 256]);
		let long_vec2 = DBVec::from_u32_slice(&[256, 256, 256, 256, 256, 256, 256, 256, 256, 257, 256, 256]);
		let result_2 = long_vec1.longest_common_prefix(&long_vec2);
		assert_eq!(long_vec1, result_2);
		let result_3 = long_vec2.longest_common_prefix(&long_vec1);
		assert_eq!(long_vec1, result_3);
		let long_vec3 = DBVec::from_u32_slice(&[256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256, 256]);
		let result_4 = long_vec2.longest_common_prefix(&long_vec3);
		assert_eq!(long_vec1, result_4);
	}

	#[test]
	fn different_suffix() {
		let vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		let (bit, suffix1) = vec1.different_suffix(30);
		println!("suffix1: {:?}", suffix1);
		assert_eq!(suffix1, DBVec::from_elem(33, true));

		let vec2 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		let (bit2, suffix2) = vec2.different_suffix(31);
		println!("suffix2: {:?}", suffix2);
		assert_eq!(suffix2, DBVec::from_elem(32, true));
	}
}
