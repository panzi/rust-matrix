#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use std::ops::MulAssign;

use matrix::{*, ops::{Pow, PowAssign, Slice}, range::Range};

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

    assert_eq!(&m3x2 * &v3, [[ 7., 16., 27.], [28., 40., 54.]]);
    assert_eq!(m3x2.by_column() * &v2, [[ 7., 14., 21.], [32., 40., 48.]]);
    assert_eq!(m3x2.by_column() * &v2 - 3.0, [[ 4., 11., 18.], [29., 37., 45.]]);

    let mut m3x2b = m3x2.clone();
    m3x2b *= &v3;

    assert_eq!(m3x2b, [[ 7., 16., 27.], [28., 40., 54.]]);

    let mut m3x2b = m3x2.clone();
    m3x2b.by_column_mut().mul_assign(&v2);

    assert_eq!(m3x2b, [[ 7., 14., 21.], [32., 40., 48.]]);

    let m3x2b = m3x2.clone().into_by_column() * &v2;

    assert_eq!(m3x2b, [[ 7., 14., 21.], [32., 40., 48.]]);

    // println!("{}", Matrix::from([
    //     [100.0, 2.0,  300.0],
    //     [  4.0, 5.555,  6.666],
    // ]));
    // println!("{:#?}", m3x2);
    // assert!(false);
}

#[test]
fn slice() {
    let m = Matrix::from([
        [ 1.0,  2.0,  3.0,  4.0,  5.0],
        [ 6.0,  7.0,  8.0,  9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0],
    ]);

    assert_eq!(m.slice((&[0], &[1, 3])), Matrix::from([[6.0], [16.0]]));
    assert_eq!(m.slice((&[0, 2], 1)), Matrix::from([[6.0, 8.0]]));
    assert_eq!(m.slice((0, &[1])), Matrix::from([[6.0]]));
    assert_eq!(m.slice((1, 3)), Matrix::from([[17.0]]));
    assert_eq!(m.slice([(0, 1), (2, 3)]), Vector::from([6.0, 18.0]));
    assert_eq!(
        m.slice([[(1, 3), (2, 3)], [(0, 1), (1, 3)]]),
        Matrix::from([[17.0, 18.0], [6.0, 17.0]]));

    assert_eq!(m.slice((&[1, 3], m.range_y())), Matrix::from([
        [ 2.0,  4.0],
        [ 7.0,  9.0],
        [12.0, 14.0],
        [17.0, 19.0],
    ]));

    assert_eq!(m.slice((Range::<2, 5>(), Range::<1, 3>())), Matrix::from([
        [ 8.0,  9.0, 10.0],
        [13.0, 14.0, 15.0],
    ]));

    assert_eq!(m.slice(m.range_xy()), m.clone());

    assert_eq!(m.slice((m.range_x(), 1)), Matrix::from([
        [ 6.0,  7.0,  8.0,  9.0, 10.0],
    ]));
}

// TODO: many more
