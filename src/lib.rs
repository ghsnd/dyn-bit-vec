extern crate bit_field;
extern crate rayon;
use std::vec::Vec;
use std::fmt;
use self::bit_field::BitField;
use self::rayon::prelude::*;
use std::u32::MAX;
use std::cmp;

#[derive(Clone, Eq, PartialEq, Hash)]
pub struct DBVec {
	words: Vec<u32>,
	cur_bit_index: u8	// current bit index in the last word.
						// 0: first bit
						// 0 and words.len() == 0: empty
						// 31: last bit
}

impl DBVec {

	///// constructors /////

	pub fn new() -> Self {
		DBVec {
			words: Vec::new(),
			cur_bit_index: 0,
		}
	}

	pub fn from_u32_slice(slice: &[u32]) -> Self {
		let mut temp_vec = Vec::with_capacity(slice.len());
		temp_vec.extend_from_slice(slice);
		DBVec {
			words: temp_vec,
			cur_bit_index: 31,
		}
	}

	pub fn from_bytes(bytes: &[u8]) -> Self {
		let mut temp_vec: Vec<u32> = Vec::with_capacity(bytes.len() * 4);
		let mut temp_int = 0u32;
		for byte_pos in bytes.iter().enumerate() {
			temp_int = temp_int | (*byte_pos.1 as u32) << byte_pos.0 % 4 * 8;
			if byte_pos.0 % 4 == 3 {
				temp_vec.push(temp_int);
				temp_int = 0;
			}
		}
		if bytes.len() % 4 != 0 {
			temp_vec.push(temp_int);
		}
		DBVec {
			words: temp_vec,
			cur_bit_index: ((bytes.len() * 8 - 1) % 32) as u8,
		}
	}

	pub fn from_elem(nbits: u64, bit: bool) -> Self {
		if nbits == 0 {
			return DBVec::new()
		}

		let elem = match bit {
			false => 0,
			true  => MAX
		};
		let len = nbits / 32 + 1;
		let rem = nbits % 32;
		let mut word_vec = vec![elem; len as usize];
		if rem > 0 {
			if let Some(last_word) = word_vec.last_mut() {
				*last_word = *last_word >> (32 - rem);
			}
		} else {
			word_vec.pop();
		}
		DBVec {
			words: word_vec,
			cur_bit_index: ((31 + rem) % 32) as u8
		}
	}

	pub fn copy(&self) -> Self {
		let mut new_vec = Vec::with_capacity(self.words.len());
		new_vec.extend(self.words.iter());
		DBVec {
			words: new_vec,
			cur_bit_index: self.cur_bit_index
		}
	}
	////////////////////////

	pub fn words(&self) -> &Vec<u32> {
		&self.words
	}

	pub fn len(&self) -> u64 {
		match self.words.len() {
			0 => 0,
			_ => ((self.words.len() - 1) * 32) as u64 + self.cur_bit_index as u64 + 1
		}
	}

	pub fn to_bytes(&self) -> Vec<u8> {
		let len = self.len();
		let mut byte_vec: Vec<u8> = Vec::with_capacity(self.words.len() / 8);
		let mut bit_counter = 0;
		for word in &self.words {
			let mut temp_word = 0 + word; // this is a lame trick to fight the borrow checker... can be done better?
			for _ in 0..4 {
				if bit_counter >= len {
					break;
				}
				let byte = (temp_word & 0b00000000_00000000_00000000_11111111) as u8;
				byte_vec.push(byte);
				temp_word = temp_word >> 8;
				bit_counter += 8;
			}
		}
		byte_vec
	}

	fn inc_len(&mut self) {
		if self.words.is_empty() {
			self.words.push(0);
			self.cur_bit_index = 0;
		} else {
			self.cur_bit_index += 1;
			if self.cur_bit_index == 32 {
				self.words.push(0);
				self.cur_bit_index = 0;
			}
		}
	}

	pub fn is_empty(&self) -> bool {
		self.len() == 0
	}

	// count ones *before* index.
	pub fn rank_one(&self, index: u64) -> u64 {
		// count ones in all words but last
		let words_index = (index / 32) as usize;
		let words_part = &self.words[..words_index];
		let mut nr_bits = match index > 32000000 {
			false => words_part
						.iter()
						.fold(0, |nr_bits, word| nr_bits + word.count_ones() as u64),
			true  => words_part
						.par_iter()
						.map(|word| word.count_ones() as u64)
						.sum()
		};

		// count ones until index in last word
		let bit_index = (index % 32) as usize;
		if bit_index != 0 {
			if let &Some (last_word) = &self.words.last() {
				let mask = !(MAX << bit_index);
				let relevant_bits = mask & last_word;
				nr_bits += relevant_bits.count_ones() as u64;
			}
		}
		nr_bits
	}

	pub fn rank_zero(&self, pos: u64) -> u64 {
		if pos == 0 {
			pos
		} else {
			pos - self.rank_one(pos)
		}
	}

	pub fn rank(&self, bit: bool, pos: u64) -> u64 {
		match bit {
			false => self.rank_zero(pos),
			true => self.rank_one(pos)
		}
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

	pub fn set_none(&mut self) {
		for word in &mut self.words {
			*word = 0;
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

		let mut last_bit = 0;
		// change the word that has to be changed
		if let Some(word) = self.words.get_mut(word_index) {
			last_bit = *word & 0b10000000_00000000_00000000_00000000u32;
			Self::insert_in_word(word, bit_index, bit);
		} 

		// for every word from word_index + 1 until end: shift left; put last_bit as first bit; remember last_bit etc
		let word_iter = self.words.iter_mut().skip(word_index + 1);
		for word in word_iter {
			let first_bit = last_bit >> 31;
			last_bit = *word & 0b10000000_00000000_00000000_00000000u32;
			*word = *word << 1;
			*word |= first_bit;
		}
	}

	// push a bit to the end. This can slightly more efficient than insert(bit, len())
	// because insert requires additional checks
	pub fn push(&mut self, bit: bool) {
		self.inc_len();
		if bit {
			let word_index = self.words.len() - 1 as usize;
			if let Some(word) = self.words.get_mut(word_index) {
				word.set_bit(self.cur_bit_index as usize, bit);
			}
		}
	}

	pub fn delete(&mut self, index: u64) {
		if index > self.len() {
			panic!("Index out of bounds: index = {} while the length is {}", index, self.len());
		}
		let bit_index = (index % 32) as usize;
		let word_index = (index / 32) as usize;

		let mut first_bit = 0;
		// for every word from end until word_index + 1: shift right; put first_bit as last bit; remember first_bit etc
		for word in self.words.iter_mut().skip(word_index + 1).rev() {
			let last_bit = first_bit << 31;
			first_bit = *word & 0b00000000_00000000_00000000_00000001u32;
			*word = *word >> 1;
			*word |= last_bit;
		}
		println!("self1: {:?}", self);

		// delete the relevant bit
		if let Some(word) = self.words.get_mut(word_index) {
			Self::delete_from_word(word, bit_index);
			*word |= first_bit << 31;
		}

		println!("self2: {:?}", self);
		// decrease length and check if the last word is to be deleted
		if self.cur_bit_index == 0 {
			self.words.pop();
			if !self.words.is_empty() {
				self.cur_bit_index = 31;
			}
		} else {
			self.cur_bit_index -= 1;
		}
	}

	// Position (index) of occurrence_nr-th occurrence of bit. Starts at one!
	pub fn select(&self, bit: bool, occurrence_nr: u64) -> Option<u64> {
		if occurrence_nr == 0 {
			return None
		}
		let mut count = 0;
		let mut prev_count = 0;

		for (index, word) in self.words.iter().enumerate() {
			// it would be better if this check on calling which function could happen *before* the iteration
			count += match bit {
				false => word.count_zeros() as u64,
				true  => word.count_ones() as u64
			};
			if count >= occurrence_nr {
				for bit_index in 0..32 {
					let word_bit = word.get_bit(bit_index);
					if word_bit == bit {
						prev_count += 1;
					}
					if prev_count == occurrence_nr {
						let result_index = (index * 32 + bit_index) as u64;
						if result_index < self.len() {
							return Some(result_index)
						} else {
							return None;
						}
					}
				}
			} else {
				prev_count = count;
			}
		}
		None
	}

	pub fn append_vec(&mut self, other: &mut Self) {
		let self_len = self.len();
		self.cur_bit_index = match self_len % 32 {
			0 => other.cur_bit_index,
			_ => {
					let sum_len = self_len + other.len();
					let bits_to_align = (self_len % 32) as u8;
					other.align_to_end(bits_to_align);
					let mut pop_self = false;
					if let Some(first_of_other) = other.words.first_mut() {
						if let Some(last_of_self) = self.words.last() {
							*first_of_other = *last_of_self | *first_of_other;
							pop_self = true;
						}
					}
					if pop_self {
						self.words.pop();
					}
					match sum_len {
						0 => 0,
						_ => ((sum_len - 1) % 32) as u8
					}
				}
		};
		self.words.append(&mut other.words);
	}

//	pub fn insert_vec(&mut self, other: &mut Self, index: u64) {
//		let self_len = self.len();
//		if index > self_len {
//			panic!("Index out of bound: index = {} while the length is {}", index, self_len);
//		}
//
//		let new_cur_bit_index = (self.cur_bit_index + other.cur_bit_index + 1) % 32;
//		// determine insertion point
//		let start_insertion_bit_index = (index % 32) as u8;
//		let end_insertion_bit_index = (start_insertion_bit_index + other.cur_bit_index + 1) % 32;
//		other.align_to_end(start_insertion_bit_index);
//		let mut self_tail_vec = self.split(index);
//		if !self_tail_vec.is_empty() {
//			if start_insertion_bit_index < end_insertion_bit_index {
//				let shift_amount = end_insertion_bit_index - start_insertion_bit_index;
//				self_tail_vec.align_to_end(shift_amount);
//			} else if start_insertion_bit_index > end_insertion_bit_index{
//				let shift_amount = start_insertion_bit_index - end_insertion_bit_index;
//				self_tail_vec.shift_to_begin(shift_amount);
//				self_tail_vec.words.pop();
//			}
//		}
//
//		// 'merge' last word of first part of self with first word of other
//		if let Some(last_of_first_part) = self.words.last() {
//			if let Some(first_other) = other.words.first_mut() {
//				*first_other = *last_of_first_part | *first_other;
//			}
//		}
//		self.words.pop();
//
//		// 'merge' last word of other with first word of last part of self_tail_vec
//		if let Some(last_of_other) = other.words.last() {
//			if !self_tail_vec.is_empty() {
//				if let Some(first_tail) = self_tail_vec.words.first_mut() {
//					*first_tail = *last_of_other | *first_tail;
//				}
//			} else {
//				self_tail_vec.words.push(*last_of_other);
//			}
//		}
//		other.words.pop();
//
//		//merge vectors
//		self.words.append(&mut other.words);
//		self.words.append(&mut self_tail_vec.words);
//		self.cur_bit_index = new_cur_bit_index;
//	}

	// insert a bit in a given word at index bit_index. The bits after bit_index shift one place towards the end
	#[inline]
	fn insert_in_word(word: &mut u32, bit_index: usize, bit: bool) {
		let shifted_word = ((MAX << bit_index) & *word) << 1;
		*word &= MAX >> (31 - bit_index);
		*word = match bit {
			false => shifted_word | *word,
			true  => {
				let word_with_bit_set = 1 << bit_index;
				shifted_word | word_with_bit_set | *word
			}
		};
	}

	// delete a bit from a given word at index bit_index. The bits after bit_index shift one place towards the beginning
	#[inline]
	fn delete_from_word(word: &mut u32, bit_index: usize) {
		match bit_index {
			0  => *word = *word >> 1,
			31 => *word = *word & 0b01111111_11111111_11111111_11111111u32,
			_  => {
				let shifted_word = ((MAX << bit_index + 1) & *word) >> 1;
				println!("+++ shifted_word :  {:032b}", shifted_word);
				let remaining_word = (MAX >> (32 - bit_index)) & *word;
				println!("+++ remaining_word: {:032b}", remaining_word);
				*word = shifted_word | remaining_word;
			}
		}
		println!("+++ word after d: {:032b}", word);
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
				let mask = MAX >> (31 - other.cur_bit_index);
				let self_last_bits = self.words.get(common_word_len).unwrap() & mask;
				let other_last_bits = other.words.last().unwrap() & mask;
				self_last_bits == other_last_bits
			} else {
				false
			}
		}
	}

	// Shifts everything nr_bits (max 31 bits) towards the end of the vector.
	// This means nr_bits leading zero's are introduced; the vector grows.
	// Overflowing bits are put into a new word at the end of the vector.
	pub fn align_to_end(&mut self, nr_bits: u8) {
		if nr_bits == 0 {
			return;
		}
		/*if self.cur_bit_index == 31 {
			self.words.push(0u32);
		}*/
		let overflowing_bits = (MAX >> nr_bits) ^ MAX;

		// check if next word needed? self.cur_bit_index + nr_bits > 32 ???
		self.cur_bit_index += nr_bits;
		if self.cur_bit_index > 31 {
			self.words.push(0u32);
			self.cur_bit_index = self.cur_bit_index % 32;
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
			if self.cur_bit_index == 0 {
				self.cur_bit_index = 32;
			} else if self.cur_bit_index < nr_bits {
				self.cur_bit_index += 32;
				self.words.pop();
			}
			self.cur_bit_index = (self.cur_bit_index - nr_bits) % 32;
		}
	}

	// split the vector at index 'at'. DOES NOT ALIGN SECOND PART!!
	pub fn split(&mut self, at: u64) -> Self {
		if at == self.len() {
			return DBVec::new();
		}

		// just split the words vector
		let at_word = at / 32;
		let mut other_words = self.words.split_off(at_word as usize);
		self.words.shrink_to_fit();

		// put the first relevant bits of other_words at the end of self.words
		let start_insertion_bit_index = (at % 32) as u8;
		if start_insertion_bit_index != 0 {
		let other_bit_mask = MAX << start_insertion_bit_index;
			let self_bit_mask = other_bit_mask ^ MAX;
			if let Some(first_of_other) = other_words.first_mut() {
				let last_of_self = *first_of_other & self_bit_mask;
				*first_of_other = *first_of_other & other_bit_mask;
				self.words.push(last_of_self);
			}
		}
		DBVec {
			words: other_words,
			cur_bit_index: 0
		}
	}

	// returns the longest common prefix of self and the other
	pub fn longest_common_prefix (&self, other: &DBVec) -> DBVec {
		let zipped_iter = self.words.iter().zip(other.words.iter());
		let mut common_words: Vec<u32> = zipped_iter
			.take_while(|&(word_1, word_2)| word_1 == word_2)
			.map(|(word_1, _word_2)| *word_1)
			.collect();

		// now check next words, if any
		let smallest_size = cmp::min(self.len(), other.len());
		let mut processed_bits = 32 * common_words.len() as u64;
		if processed_bits < smallest_size {
			// check next word
			let index = (processed_bits / 32) as usize;
			let word_1 = self.words.get(index).unwrap();
			let word_2 = other.words.get(index).unwrap();
			let mut bits_to_check = (smallest_size - processed_bits) as usize;
			if bits_to_check >= 32 {
				bits_to_check = 31;	// to avoid overflow of shift when calculating to_check_mask
			}
			if bits_to_check > 0 {
				// now determine how many bits are the same
				// example:
				// word_1 =             1100
				// word_2 =             1000
				// nr_bits_to_check = 3
				// word_1 XOR word_2 =  0100   => bit on pos 2 differs! This is equal to the the number of trailing zero's.
				// the to_check_mask is there to set bits beyond bits_to_check zero and thus to ignore them.
				let to_check_mask = MAX << bits_to_check;
				let diff_bit_pos_mask = word_1 ^ word_2 ^ to_check_mask;
				let bit_nr = diff_bit_pos_mask.trailing_zeros() as usize;
				// now bit_nr bits are the same.
				if bit_nr > 0 {
					let bits_too_much = 32 - bit_nr;
					let mask = MAX >> bits_too_much;
					let result_word = word_1 & mask;
					common_words.push(result_word);
					processed_bits += bit_nr as u64;
				}
			}
		} else {
			// set bits that exceed the size to 0
			if let Some(last_word) = common_words.last_mut() {
				let bits_too_much = processed_bits - smallest_size;
				let mut mask = MAX >> bits_too_much;
				*last_word = *last_word & mask;
				processed_bits = smallest_size;
			}
		}
		DBVec {
			words: common_words,
			cur_bit_index: match processed_bits {
				0 => 0,
				_ => ((processed_bits - 1) % 32) as u8
			}
		}
	}

	pub fn different_suffix(&self, at: u64) -> (bool, Self) {
		let first_bit = self.get(at);
		let new_at = at + 1;
		let at_word = (new_at / 32) as usize;
		let at_bit = (new_at % 32) as u8;
		let mut result_vec = DBVec::from_u32_slice(&self.words[at_word..]);
		result_vec.cur_bit_index = self.cur_bit_index;
		result_vec.shift_to_begin(at_bit);	// the shift corrects cur_bit_index
		(first_bit, result_vec)
	}

}

impl fmt::Debug for DBVec {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		let len = self.len();
		let _a = write!(f, "DBVec: ({}, {}, ", len, self.rank_one(len));
		let mut count = 0u8;
		for word in self.words.iter() {
			let _b = write!(f, "{:032b} ", word);
			count += 1;
			if count == 100 {
				count = 1;
				let _c = write!(f, "\n");
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
		let len = vec.len();
		assert_eq!(len, 96);
		assert_eq!(vec.rank_one(len), 3);
	}

	#[test]
	fn from_elem() {
		let mut vec1 = DBVec::from_elem(30, false);
		println!("{:?}", vec1);
		let mut vec2 = DBVec::from_elem(35, true);
		println!("{:?}", vec2);
		let mut vec3 = DBVec::from_elem(5, false);
		vec1.append_vec(&mut vec2);
		vec1.append_vec(&mut vec3);
		println!("{:?}", vec1);
		assert_eq!(vec1.len(), 70);
		assert_eq!(vec1.words(), &[0b11000000000000000000000000000000, 0b11111111111111111111111111111111, 0b00000000000000000000000000000001]);

		let vec_test = DBVec::from_elem(32, true);
		assert_eq!(DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]), vec_test);
	}

	#[test]
	fn insert_bit() {
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
		let len = vec.len();
		assert_eq!(48, len);
		assert_eq!(11, vec.rank_one(len));
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
		let vec7 = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000]);
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
	fn split_at_length() {
		// split at length => tail should be empty
		let mut short_vec = DBVec::new();
		short_vec.push(false);
		short_vec.push(true);
		println!("{:?}", short_vec);
		let tail_vec = short_vec.split(2);
		println!("tail: {:?}", tail_vec);
		assert!(tail_vec.is_empty());
		assert_eq!(short_vec.len(), 2);
		assert_eq!(short_vec.rank_one(2), 1);
	}

	#[test]
	fn split_two_in_half() {
		// split a vector of 2 words into 2 of one word
		let mut vec = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		println!("{:?}", vec);
		let tail_vec = vec.split(32);
		println!("tail: {:?}", tail_vec);
		assert_eq!(vec.words, &[0b11111111_11111111_11111111_11111111u32]);
		assert_eq!(tail_vec.words, &[0b11111111_11111111_11111111_11111111u32]);
	}

	#[test]
	fn split_somewhere_else() {
		// split a vector of 2 words somewhere in the second word
		let mut vec = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		println!("{:?}", vec);
		let tail_vec = vec.split(34);
		println!("tail: {:?}", tail_vec);
		assert_eq!(tail_vec.words, &[0b11111111_11111111_11111111_11111100u32]);
	}

//	#[test]
//	fn insert_vec() {
//		let mut vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
//		let mut vec2 = DBVec::from_bytes(&[0b01111110]);
//		//println!("vec1: {:?}", vec1);
//		//println!("vec2: {:?}", vec2);
//		vec1.insert_vec(&mut vec2, 4);
//		println!("vec1: {:?}", vec1);
//		//println!("vec2: {:?}", vec2);
//		assert_eq!(vec1.words(), &[0b11111111111111111111011111101111, 0b00000000000000000000000011111111]);
//		let mut vec3 = DBVec::from_u32_slice(&[256, 256, 256, 256, 256, 256, 256, 256, 256]);
//		println!("vec3: {:?}", vec3);
//		vec1.insert_vec(&mut vec3, 34);
//		println!("vec1: {:?}", vec1);
//		assert_eq!(vec1.len(), 328);
//		assert_eq!(vec1.words(), &[0b11111111111111111111011111101111, 0b00000000000000000000010000000011,
//			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
//			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
//			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
//			0b00000000000000000000010000000000, 0b00000000000000000000010000000000,
//			0b00000000000000000000000011111100]);
//	}

	#[test]
	fn append_vec_short() {
		let mut vec1 = DBVec::new();
		vec1.push(true);
		vec1.push(false);
		vec1.push(true);
		let mut vec2 = DBVec::new();
		vec2.push(true);
		vec2.push(false);
		vec2.push(true);
		vec1.append_vec(&mut vec2);
		let mut result = DBVec::new();
		result.push(true);
		result.push(false);
		result.push(true);
		result.push(true);
		result.push(false);
		result.push(true);
		assert_eq!(vec1, result);
	}

	#[test]
	fn append_vec_border_cases() {
		// empty
		let mut vec1 = DBVec::new();
		let mut vec2 = DBVec::new();
		vec1.append_vec(&mut vec2);
		assert_eq!(vec1.len(), 0);

		// append 1 to empty
		let mut vec3 = DBVec::new();
		let mut vec4 = DBVec::new();
		vec4.push(true);
		vec3.append_vec(&mut vec4);
		assert_eq!(vec3.len(), 1);
		assert_eq!(vec3.words, vec![1]);
		assert_eq!(vec4.len(), 0);

		// test a lot
		for nr_bits in 0..1000 {
			println!("nr_bits: {}", nr_bits);
			let mut vec_a = DBVec::new();
			let mut vec_b = DBVec::new();
			for _ in 0..nr_bits {
				vec_a.push(true);
				vec_b.push(true);
			}
			vec_a.append_vec(&mut vec_b);
			assert_eq!(vec_a.len(), 2 * nr_bits);
			assert_eq!(vec_a.rank_one(2 * nr_bits), 2 * nr_bits);
		}
	}

	#[test]
	fn longest_common_prefix() {
		// simple one-word vectors
		let vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		let vec2 = DBVec::from_u32_slice(&[0b11111111_11111111_11111011_11111111u32]);
		let exp  = DBVec {
			words: vec!(0b11_11111111u32),
			cur_bit_index: 9
		};
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
	fn longest_common_prefix_2() {
		let vec1 = DBVec::from_u32_slice(&[0b00110010_00010111_10010111_10111000]);
		let vec2 = DBVec::from_u32_slice(&[0b00010000_00010001_00110111_10111000]);
		let result = vec1.longest_common_prefix(&vec2);
		let exp  = DBVec {
			words: vec!(0b10111_10111000u32),
			cur_bit_index: 12
		};
		println!("{:?}", result);
		assert_eq!(exp, result);
	}

	#[test]
	fn longest_common_prefix_3() {
		//DBVec: (1, 0, 00000000000000000000000000000000 )
		//DBVec: (5, 1, 00000000000000000000000000000100 )
		//DBVec: (0, 0, )
		let mut vec1 = DBVec::new();
		vec1.push(false);
		let mut vec2 = DBVec::new();
		vec2.push(false);
		vec2.push(false);
		vec2.push(true);
		vec2.push(false);
		vec2.push(false);
		let lcp = vec1.longest_common_prefix(&vec2);
		println!("v1: {:?}\nv2: {:?}\nr : {:?}", vec1, vec2, lcp);
		let exp = DBVec {
			words: vec!(0),
			cur_bit_index: 0
		};
		assert_eq!(lcp, exp);
	}

	#[test]
	fn different_suffix() {
		let vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		let (bit, suffix1) = vec1.different_suffix(30);
		println!("suffix1: {:?}", suffix1);
		assert_eq!(bit, true);
		assert_eq!(suffix1, DBVec::from_elem(33, true));

		let vec2 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32, 0b11111111_11111111_11111111_11111111u32]);
		let (bit2, suffix2) = vec2.different_suffix(31);
		println!("suffix2: {:?}", suffix2);
		assert_eq!(bit2, true);
		assert_eq!(suffix2, DBVec::from_elem(32, true));
	}

	#[test]
	fn rank() {
		let vec1 = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		assert_eq!(21, vec1.rank(true, 21));
		assert_eq!(0, vec1.rank(false, 21));
	}

	#[test]
	fn copy() {
		let vec1 = DBVec::from_u32_slice(&[1234, 56789, 10111213]);
		let vec2 = vec1.copy();
		assert_eq!(vec1, vec2);
	}

	#[test]
	fn select() {
		// pretty normal case
		let vec = DBVec::from_u32_slice(
		&[0b11111111111111111111011111101111,
		  0b11111100111111111111111111111110,
		  0b11111111111111111111011111101111,
		  0b11111111111101111111110111100011]);
		assert_eq!(vec.select(false, 1), Some(4));
		assert_eq!(vec.select(false, 2), Some(11));
		assert_eq!(vec.select(false, 12), Some(115));
		assert_eq!(vec.select(true, 1), Some(0));
		assert_eq!(vec.select(true, 2), Some(1));
		assert_eq!(vec.select(true, 31), Some(33));
		assert_eq!(vec.select(false, 0), None);

		// empty vec
		let vec2 = DBVec::new();
		assert_eq!(vec2.select(false, 1), None);
		assert_eq!(vec2.select(true, 1), None);

		// vec with 2 elements, see if we go over bandaries
		let mut vec3 = DBVec::new();
		vec3.push(false);
		vec3.push(true);
		assert_eq!(vec3.select(false, 1), Some(0));
		assert_eq!(vec3.select(true, 1), Some(1));
		assert_eq!(vec3.select(false, 2), None);
		assert_eq!(vec3.select(true, 2), None);
	}

	#[test]
	fn delete() {
		let mut vec = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		println!("initial vec: {:?}", vec);
		vec.push(true);
		println!("initial vec: {:?}", vec);
		vec.delete(5);
		let result = DBVec::from_u32_slice(&[0b11111111_11111111_11111111_11111111u32]);
		assert_eq!(vec, result);
	}

	#[test]
	fn to_bytes() {
		//let vec1 = DBVec::from_bytes(&[1]);
		//let bytes
		for nr_bytes in 1..100 {
			let mut byte_vec: Vec<u8> = Vec::with_capacity(nr_bytes);
			for byte in 0..nr_bytes {
				byte_vec.push(byte as u8);
			}
			let vec = DBVec::from_bytes(&byte_vec);
			println!("{:?}", vec);
			let byte_vec_2 = vec.to_bytes();
			assert_eq!(byte_vec, byte_vec_2);
		}
	}
}
