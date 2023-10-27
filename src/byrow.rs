use crate::{Matrix, Number};

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct IntoByRow<const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: Matrix<X, Y, T>
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ByRow<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a Matrix<X, Y, T>
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ByRowMut<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a mut Matrix<X, Y, T>
}

impl<const X: usize, const Y: usize, T: Number> IntoByRow<X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> ByRow<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> ByRowMut<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a mut Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }
}
