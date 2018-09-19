# dyn_bit_vec


This crate contains a bit vector that supports only operations used in
[Wavelet trie](https://github.com/ghsnd/wavelet-trie). Other operations
might be added in the future, but it is recommended to use [other bit vector implementations](https://crates.io/search?q=bitvec).

### Bit operations supported:

* delete
* get
* insert
* push
* rank
* select

### Vector operations supported:

* starts_with
* longest_common_prefix
* append
* split

**WARNING:** this is work in progress; it may contain bugs!

# Examples

Have a look at the tests :)
