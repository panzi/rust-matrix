use crate::{Matrix, Number, Vector, FromUSize, iter::ColumnIter};

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct IntoByColumn<const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: Matrix<X, Y, T>
}

#[repr(transparent)]
#[derive(Debug, Clone)]
pub struct ByColumn<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a Matrix<X, Y, T>
}

#[repr(transparent)]
#[derive(Debug)]
pub struct ByColumnMut<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a mut Matrix<X, Y, T>
}

impl<const X: usize, const Y: usize, T: Number> IntoByColumn<X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }

    pub fn iter_vectors(&self) -> ColumnIter<'_, X, Y, T> {
        self.matrix.columns()
    }

    #[inline]
    pub fn map<F, U>(self, f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        self.matrix.transpose_map(f)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }

    pub fn iter_vectors(&self) -> ColumnIter<'_, X, Y, T> {
        self.matrix.columns()
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        self.matrix.transpose_map(f)
    }

    #[inline]
    pub fn fold<F, B>(&self, init: B, mut f: F) -> Vector<X, B>
    where F: FnMut(B, T) -> B, B: Number {
        let mtx = self.matrix.data();
        let mut data = Box::new([init; X]);

        for y in 0..Y {
            let yoffset = y * X;
            for (value, res) in mtx[yoffset..yoffset + X].iter().zip(data.iter_mut()) {
                *res = f(*res, *value);
            }
        }

        Vector::from(data)
    }

    #[inline]
    pub fn sum(&self) -> Vector<X, T> {
        self.fold(T::default(), |acc, value| acc + value)
    }

    #[inline]
    pub fn avg(&self) -> Vector<X, T>
    where T: FromUSize {
        self.sum() / T::from_usize(Y)
    }

    #[inline]
    pub fn mean(&self) -> Vector<X, T>
    where T: Ord, [T; Y * X]: Sized {
        self.matrix.transpose().mean()
    }
}

// TODO: impl Slice

impl<'a, const X: usize, const Y: usize, T: Number> ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a mut Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }

    pub fn iter_vectors(&'a self) -> ColumnIter<'a, X, Y, T> {
        self.matrix.columns()
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        self.matrix.transpose_map(f)
    }
}
