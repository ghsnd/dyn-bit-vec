mod dyn_bit_vec {

extern crate bit_field;
use std::vec::Vec;
use std::fmt;
use self::bit_field::BitField;

pub struct DBVec {
	words: Vec<u32>,
	len: u64,
	popcnt: u64
}

impl DBVec {
	pub fn new() -> Self {
		DBVec {
			words: Vec::new(),
			len: 0,
			popcnt: 0
		}
	}

	pub fn from_u32_slice(slice: &[u32]) -> Self {
		let mut temp_vec = Vec::with_capacity(slice.len());
		temp_vec.extend_from_slice(slice);
		let bit_count = slice.iter().fold(0, |nr_bits, number| nr_bits + number.count_ones());
		DBVec {
			words: temp_vec,
			len: slice.len() as u64 * 32,
			popcnt: bit_count as u64
		}
	}

	pub fn len(&self) -> u64 {
		self.len
	}

	// insert a bit at position 'index'
	pub fn insert(&mut self, bit: bool, index: u64) {
		if index > self.len {
			panic!("Index out of bound: index = {} while the length is {}", index, self.len);
		}
		if self.len % 32 == 0 {
			self.words.push(0);
		}
		self.len += 1;
		if bit {
			self.popcnt += 1;
		}
		let bit_index = (index % 32) as u8;
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
		let bit_index = (self.len % 32) as u8;
		if bit_index == 0 {
			self.words.push(0);
		}
		self.len += 1;
		if bit {
			self.popcnt += 1;
			let word_index = (self.len / 32) as usize;
			if let Some(word) = self.words.get_mut(word_index) {
				word.set_bit(bit_index, bit);
			}
		}
	}

	// insert a bit in a given word at index bit_index. The bits after bit_index shift one place towards the end
	#[inline]
	fn insert_in_word(word: &mut u32, bit_index: u8, bit: bool) {
		for remaining_bit_index in (bit_index + 1..32).rev() {
			let prev_bit = word.get_bit(remaining_bit_index - 1);
			word.set_bit(remaining_bit_index, prev_bit);
		}
		word.set_bit(bit_index, bit);
	}

}

impl fmt::Debug for DBVec {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "DBVec: ({}, {}, ", self.len, self.popcnt);
		for word in self.words.iter().rev() {
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
	fn it_works() {
		assert_eq!(2 + 2, 4);
		let int_1 = 0b01u32;
		println!("int_1: {:032b} - {}", int_1, int_1);
		let int_2 = int_1 << 9;
		println!("int_1 << 9: {:032b} - {}", int_2, int_2);
	}

	#[test]
	fn from_u32_slice() {
		let DBVec = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		println!("{:?}", DBVec);
	}

	#[test]
	fn insert() {
		let mut DBVec = DBVec::new();
		DBVec.insert(true, 0);
		println!("{:?}", DBVec);
		for _ in 1..42 {
			DBVec.insert(true, 1);
			println!("{:?}", DBVec);
		}
		DBVec.insert(false, 42);
		DBVec.insert(true, 43);
		println!("{:?}", DBVec);
		for c in 42..62 {
			DBVec.insert(true, c);
			println!("{:?}", DBVec);
		}
	}

	#[test]
	fn push() {
		let mut DBVec = DBVec::new();
		let mut bit = true;
		for _ in 1..42 {
			DBVec.push(bit);
			bit = !bit;
			println!("{:?}", DBVec);
		}
	}

	#[test]
	fn overflow() {
		let mut DBVec = DBVec::from_u32_slice(&[256; 2047]);
		//println!("{:?}", DBVec);
		for _ in 0..32 {
			DBVec.push(true);
		}
		println!("{:?}", DBVec);
	}

}
