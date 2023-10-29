#![allow(incomplete_features)]
#![feature(generic_const_exprs)]
use std::ops::MulAssign;

use matrix::{*, ops::{Pow, PowAssign, Slice, Unit}, range::{Range, RangeIter}};

#[test]
fn unit() {
    let m = Matrix::unit();

    assert_eq!(&m, &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);

    assert_eq!(m, [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);

    assert_eq!(m, &[
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]);

    let v = Vector::unit();

    assert_eq!(&v, &[1.0, 1.0, 1.0]);
    assert_eq!(v,   [1.0, 1.0, 1.0]);
    assert_eq!(v,  &[1.0, 1.0, 1.0]);
}

#[test]
fn transpose() {
    let m = Matrix::from([
        [1, 2, 3],
        [4, 5, 6],
    ]);

    assert_eq!(m.transpose(), [
        [1, 4],
        [2, 5],
        [3, 6],
    ]);

    let mut m = Matrix::from([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]);

    assert_eq!(m.transpose(), [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]);

    m.transpose_assign();

    assert_eq!(m, [
        [1, 4, 7],
        [2, 5, 8],
        [3, 6, 9],
    ]);

    let m = Matrix::from([[0usize; 0]; 0]);
//    assert_eq!(m.clone().into_transpose(), m.transpose());
    let mut m2 = m.clone();
    m2.transpose_assign();
    assert_eq!(m2, m.transpose());

    let m = Matrix::from([[0usize]]);
    let mut m2 = m.clone();
    m2.transpose_assign();
//    assert_eq!(m.clone().into_transpose(), m.transpose());
    assert_eq!(m2, m.transpose());

    macro_rules! test_transpose {
        () => {};

        (@opt $x:literal) => {};
        (@opt $x:literal sym) => {
            let m: Matrix<$x, $x, usize> = Matrix::from(Range::<0, { $x * $x }>().to_vector());
            let mut m2 = m.clone();
            m2.transpose_assign();
//            println!("transpose_assign {} x {}", $x, $x);
            assert_eq!(m2, m.transpose());
            assert_eq!(m.clone().into_transpose(), m.transpose());
        };

        (($x:literal $y:literal $($opt:ident)?) $($tail:tt)*) => {
//            let v = Range::<0, { $x * $y }>().to_vector();
//            let m: Matrix<$x, $y, usize> = Matrix::from(v);
            test_transpose!(@opt $x $($opt)?);

//            println!("into_transpose {} x {}", $x, $y);
//            assert_eq!(m.clone().into_transpose(), m.transpose());
            test_transpose!($($tail)*);
        };
    }

    test_transpose!(
        (1 2) (2 1)
        (1 3) (3 1)
        (1 4) (4 1)
        (1 5) (5 1)
        (2 2 sym)
        (2 3) (3 2)
        (2 4) (4 2)
        (2 5) (5 2)
        (3 3 sym)
        (3 4) (4 3)
        (3 5) (5 3)
        (4 4 sym)
        (4 5) (5 4)
        (5 5 sym)
    );

    /*
    into_transpose()?
    [
        [0, 1, 2],
        [3, 4, 5],
    ] == [0, 1, 2, 3, 4, 5]
    
    [
        [0, 3],
        [1, 4],
        [2, 5],
    ] == [0, 3, 1, 4, 2, 5]

    ----

    [
        [0, 1,  2,  3],
        [4, 5,  6,  7],
        [8, 9, 10, 11],
    ] == [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

    [
        [0, 4,  8],
        [1, 5,  9],
        [2, 6, 10],
        [3, 7, 11],
    ] == [0, 4, 8, 1, 5, 9, 2, 6, 10, 3, 7, 11]

    [0, 1, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11] original
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3,  7, 11] transposed

    [0, 1, 2, 3, 4, 5, 6, 7,  8, 9, 10, 11]
     ^
    [0, 4, 2, 3, 1, 5, 6, 7,  8, 9, 10, 11]
        ^        ^
    [0, 4, 8, 3, 1, 5, 6, 7,  2, 9, 10, 11]
           ^                  ^
    [0, 4, 8, 1, 3, 5, 6, 7,  2, 9, 10, 11]
              ^  ^
    [0, 4, 8, 1, 5, 3, 6, 7,  2, 9, 10, 11]
                 ^  ^
    [0, 4, 8, 1, 5, 9, 6, 7,  2, 3, 10, 11]
                    ^            ^
    [0, 4, 8, 1, 5, 9, 2, 7,  6, 3, 10, 11]
                       ^      ^
    [0, 4, 8, 1, 5, 9, 2, 6,  7, 3, 10, 11]
                          ^   ^
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3,  7, 11]
                              ^      ^
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3,  7, 11]
                                 ^
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3,  7, 11]
                                     ^
    [0, 4, 8, 1, 5, 9, 2, 6, 10, 3,  7, 11]
                                        ^

    [>0,  1,  2]  [ 0,> 4,  2]  [ 0,  4,> 8]  [ 0,  4,  8]
    [ 3,  4,  5]  [ 3,> 1,  5]  [ 3,  1,  5]  [>1,> 3,  5]
    [ 6,  7,  8]  [ 6,  7,  8]  [ 6,  7,> 2]  [ 6,  7,  2]
    [ 9, 10, 11]  [ 9, 10, 11]  [ 9, 10, 11]  [ 9, 10, 11]
    
    [ 0,  4,  8]  [ 0,  4,  8]  [ 0,  4,  8]  [ 0,  4,  8]
    [ 1, >5,> 3]  [ 1,  5,> 9]  [ 1,  5,  9]  [ 1,  5,  9]
    [ 6,  7,  2]  [ 6,  7,  2]  [>2,  7,> 6]  [ 2,> 6,> 7]
    [ 9, 10, 11]  [>3, 10, 11]  [ 3, 10, 11]  [ 3, 10, 11]

    [ 0,  4,  8]  [ 0,  4,  8]  [ 0,  4,  8]  [ 0,  4,  8]
    [ 1,  5,  9]  [ 1,  5,  9]  [ 1,  5,  9]  [ 1,  5,  9]
    [ 2,  6,>10]  [ 2,  6, 10]  [ 2,  6, 10]  [ 2,  6, 10]
    [ 3,> 7, 11]  [>3,  7, 11]  [ 3,> 7, 11]  [ 3,  7,>11]
    */
}

#[test]
fn reshape() {
    let m = Matrix::from([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ]);

    assert_eq!(m.reshape(), [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ]);

    assert_eq!(m.reshape(), [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
        [6.0],
    ]);

    assert_eq!(m.into_reshape(), [
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    ]);
}

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
    assert_eq!(m3x2.transpose() * &v2, [
        [ 7., 32.],
        [14., 40.],
        [21., 48.],
    ]);
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
fn matrix_slice() {
    let m = Matrix::from([
        [ 1.0,  2.0,  3.0,  4.0,  5.0],
        [ 6.0,  7.0,  8.0,  9.0, 10.0],
        [11.0, 12.0, 13.0, 14.0, 15.0],
        [16.0, 17.0, 18.0, 19.0, 20.0],
    ]);

    assert_eq!(m.slice((&[0], &[1, 3])), Matrix::from([[6.0], [16.0]]));
    assert_eq!(m.slice((&[0, 2], 1)), Matrix::from([[6.0, 8.0]]));
    assert_eq!(m.slice(&(0, &[1])), Matrix::from([[6.0]]));
    assert_eq!(m.slice((1, 3)), Matrix::from([[17.0]]));
    assert_eq!(m.slice(&mut (1, 3)), Matrix::from([[17.0]]));
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

#[test]
fn vector_slice() {
    let v = Vector::from([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);

    assert_eq!(v.slice(1), Vector::from([2.0]));
    assert_eq!(v.slice([0, 3]), Vector::from([1.0, 4.0]));
    assert_eq!(v.slice(&[1, 2, 3]), Vector::from([2.0, 3.0, 4.0]));
    assert_eq!(v.slice(&mut [1, 2, 3]), Vector::from([2.0, 3.0, 4.0]));
    assert_eq!(v.slice(v.range()), v.clone());
    assert_eq!(v.slice(Range::<3, 6>()), Vector::from([4.0, 5.0, 6.0]));
}

#[test]
fn format() {
    let m = Matrix::from([
        [  0.0,   1.0  ,  -2.0  ],
        [333.0,   4.444,   5.555],
        [  0.6, 777.0  ,   8.0  ],
    ]);

    assert_eq!(format!("{}", m), "\
[
    [  0.0    1.0    -2.0  ]
    [333.0    4.444   5.555]
    [  0.6  777.0     8.0  ]
]");

    assert_eq!(format!("{:#?}", m), "\
Matrix::from([
    [
        0.0,
        1.0,
        -2.0,
    ],
    [
        333.0,
        4.444,
        5.555,
    ],
    [
        0.6,
        777.0,
        8.0,
    ],
])");
}

// TODO: many more
