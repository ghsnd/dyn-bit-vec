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

	pub fn from_bytes(bytes: &[u8]) -> Self {
		let mut temp_vec: Vec<u32> = Vec::with_capacity(bytes.len() * 4);
		let mut temp_int = 0u32;
		let mut bit_count = 0u64;
		for byte_pos in bytes.iter().enumerate() {
			temp_int = temp_int | (*byte_pos.1 as u32);
			if byte_pos.0 % 4 == 3 {
				bit_count += temp_int.count_ones() as u64;
				temp_vec.push(temp_int);
				temp_int = 0;
			}
			temp_int = temp_int << 8;
			println!("{:032b}", temp_int); 
		}
		if bytes.len() % 4 != 0 {
			temp_int = temp_int >> 8;
			bit_count += temp_int.count_ones() as u64;
			temp_vec.push(temp_int);
		}
		DBVec {
			words: temp_vec,
			len: bytes.len() as u64 * 8,
			popcnt: bit_count
		}
	}

	pub fn len(&self) -> u64 {
		self.len
	}

	pub fn pop_cnt(&self) -> u64 {
		self.popcnt
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
	fn it_works() {
		assert_eq!(2 + 2, 4);
		let int_1 = 0b01u32;
		println!("int_1: {:032b} - {}", int_1, int_1);
		let int_2 = int_1 << 9;
		println!("int_1 << 9: {:032b} - {}", int_2, int_2);
	}

	#[test]
	fn from_u32_slice() {
		let vec = DBVec::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		println!("{:?}", vec);
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

/*	#[test]
	fn overflow() {
		let mut DBVec = DBVec::from_u32_slice(&[256; 2047]);
		//println!("{:?}", DBVec);
		for _ in 0..32 {
			DBVec.push(true);
		}
		println!("{:?}", DBVec);
	}*/

	#[test]
	fn from_bytes() {
		let vec = DBVec::from_bytes(&[0b10000000, 0b10000010, 0b10000100, 0b10001000, 0b10010000, 0b10100000]);
		//let vec = DBVec::from_bytes(&[0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000]);
		//let vec = DBVec::from_bytes(&[0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000, 0b10000000]);
		println!("{:?}", vec);
		assert_eq!(48, vec.len());
		assert_eq!(11, vec.pop_cnt());
	}

}
