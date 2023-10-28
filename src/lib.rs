#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
#![feature(slice_as_chunks)]

mod number;
mod vector;
mod matrix;
mod assert;
pub mod ops;
pub mod range;
pub mod bycolumn;

pub use number::*;
pub use vector::*;
pub use matrix::*;
