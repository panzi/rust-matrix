#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use matrix::{*, ops::{Pow, PowAssign}};

#[test]
fn pow() {
    let mut m: Matrix<3, 4> = 2.0.into();

    m.pow_assign(2.0);

    assert_eq!(&m, &4.0.into());
    assert_eq!((&m).pow(2.0), 16.0.into());
    assert_eq!(matrix::ops::pow(&m, 2.0), 16.0.into());
    assert_eq!(3.0.pow(&m), 81.0.into());
    assert_eq!(matrix::ops::pow(m, 2.0), 16.0.into());
}
