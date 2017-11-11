mod dyn_bit_vec {

extern crate bit_field;
use std::vec::Vec;
use std::fmt;
use self::bit_field::BitField;

pub struct Block {
	words: Vec<u32>,
	len: u16,
	popcnt: u16
}

impl Block {
	pub fn new() -> Self {
		Block {
			words: Vec::new(),
			len: 0,
			popcnt: 0
		}
	}

	pub fn from_u32_slice(slice: &[u32]) -> Self {
		let mut temp_vec = Vec::with_capacity(slice.len());
		temp_vec.extend_from_slice(slice);
		let bit_count = slice.iter().fold(0, |nr_bits, number| nr_bits + number.count_ones()) as u16;
		Block {
			words: temp_vec,
			len: slice.len() as u16 * 32,
			popcnt: bit_count
		}
	}

	pub fn len(&self) -> u16 {
		self.len
	}

	pub fn insert(&mut self, bit: bool, index: u16) {
		if index > self.len {
			panic!("Index out of bound: index = {} while the length is {}", index, self.len);
		}
		if self.len % 32 == 0 {
			self.words.push(0);
		}
		self.len += 1;
		if (bit) {
			self.popcnt += 1;
		}
		let word_index = index / 32;
		let bit_index = (index % 32) as u8;

		let usize_index = word_index as usize;
		let mut last_bit = false;
		// change the word that has to be changed
		if let Some(word) = self.words.get_mut(usize_index) {
			last_bit = word.get_bit(31);
			for remaining_bit_index in (bit_index + 1..32).rev() {
				let prev_bit = word.get_bit(remaining_bit_index - 1);
				word.set_bit(remaining_bit_index, prev_bit);
			}
			word.set_bit(bit_index, bit);
		} 
		// for every word from word_index + 1 until end: shift left; put last_bit as first bit; remember last_bit etc
		let word_iter = self.words.iter_mut().skip(usize_index + 1);
		for word in word_iter {
			let first_bit = last_bit;
			last_bit = word.get_bit(31);
			*word = *word << 1;
			*word.set_bit(0, first_bit);
		}
	}
}

impl fmt::Debug for Block {
	fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
		write!(f, "Block: ({}, {}, ", self.len, self.popcnt);
		for word in self.words.iter().rev() {
			write!(f, "{:032b} ", word);
		}
		write!(f, ")")
	}
}

}

#[cfg(test)]
mod tests {

use dyn_bit_vec::Block;

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
		let block = Block::from_u32_slice(&[0b1u32, 0b10u32, 0b10000000_00000000_00000000_00000000u32]);
		println!("{:?}", block);
	}

	#[test]
	fn insert() {
		let mut block = Block::new();
		block.insert(true, 0);
		println!("{:?}", block);
		for c in 1..42 {
			block.insert(true, 1);
			println!("{:?}", block);
		}
		block.insert(false, 42);
		block.insert(true, 43);
		println!("{:?}", block);
		for c in 42..62 {
			block.insert(true, c);
			println!("{:?}", block);
		}
	}
}
