#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use std::ops::MulAssign;

use matrix::{*, ops::{Pow, PowAssign}};

#[test]
fn pow() {
    let mut m: Matrix<3, 4> = 2.0.into();

    m.pow_assign(2.0);

    assert_eq!(&m, &Matrix::from(4.0));
    assert_eq!(&m, &[[4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0], [4.0, 4.0, 4.0]]);
    assert_eq!((&m).pow(2.0), Matrix::from(16.0));
    assert_eq!(matrix::ops::pow(&m, 2.0), Matrix::from(16.0));
    assert_eq!(3.0.pow(&m), Matrix::from(81.0));
    assert_eq!(matrix::ops::pow(m, 2.0), Matrix::from(16.0));
}

#[test]
fn matrix_x_vector() {
    let m3x2 = Matrix::from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    let v2 = Vector::from([
        7.0, 8.0
    ]);

    let v3 = Vector::from([
        7.0, 8.0, 9.0
    ]);

    println!("{:?}", &m3x2 * &v2);
    println!("{:?}", m3x2.by_row() * v3.clone());

    assert_eq!(&m3x2 * &v2, [[ 7., 14., 21.], [32., 40., 48.]]);
    assert_eq!(m3x2.by_row() * &v3, [[ 7., 16., 27.], [28., 40., 54.]]);
    assert_eq!(m3x2.by_row() * &v3 - 3.0, [[ 4., 13., 24.], [25., 37., 51.]]);

    let mut m3x2b = m3x2.clone();
    m3x2b *= &v2;

    assert_eq!(m3x2b, [[ 7., 14., 21.], [32., 40., 48.]]);

    let mut m3x2b = m3x2.clone();
    m3x2b.by_row_mut().mul_assign(&v3);

    assert_eq!(m3x2b, [[ 7., 16., 27.], [28., 40., 54.]]);

    let m3x2b = m3x2.clone().into_by_row() * &v3;

    assert_eq!(m3x2b, [[ 7., 16., 27.], [28., 40., 54.]]);
}

// TODO: many more
// TODO: make by_row the default and add by_column methods instead (numpy has by_row)
