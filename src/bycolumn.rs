use std::ops::{Index, IndexMut};

use crate::{Matrix, Number, Vector, iter::ColumnIter, ops::{MatrixAggregate, Get, GetMut, Slice}, range::RangeIter};

// TODO: impl Slice

// ======== IntoByColumn =======================================================

#[repr(transparent)]
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct IntoByColumn<const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: Matrix<X, Y, T>
}

impl<const X: usize, const Y: usize, T: Number> IntoByColumn<X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: Matrix<X, Y, T>) -> Self {
        Self { matrix }
    }

    #[inline]
    pub fn iter_vectors(&self) -> ColumnIter<'_, X, Y, T> {
        self.matrix.columns()
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        self.matrix.transpose_map(f)
    }

    #[inline]
    pub fn to_matrix(&self) -> Matrix<Y, X, T>
    where [T; Y * X]: Sized {
        self.matrix.transpose()
    }
}

impl<const X: usize, const Y: usize, T: Number> MatrixAggregate<Y, X, T> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    fn fold<F, B>(&self, init: B, f: F) -> Vector<X, B>
    where F: FnMut(B, T) -> B, B: Number {
        fold_by_column(&self.matrix, init, f)
    }

    #[inline]
    fn mean(&self) -> Vector<X, T>
    where T: Ord, [T; X * Y]: Sized {
        mean_by_column(&self.matrix)
    }
}

impl<const X: usize, const Y: usize, T: Number> Get<(usize, usize)> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn get(&self, (x, y): (usize, usize)) -> Option<&Self::Output> {
        self.matrix.get(y, x)
    }
}

impl<const X: usize, const Y: usize, T: Number> GetMut<(usize, usize)> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn get_mut(&mut self, (x, y): (usize, usize)) -> Option<&mut Self::Output> {
        self.matrix.get_mut(y, x)
    }
}

impl<const X: usize, const Y: usize, T: Number> Index<(usize, usize)> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.matrix.index(y, x)
    }
}

impl<const X: usize, const Y: usize, T: Number> IndexMut<(usize, usize)> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.matrix.index_mut(y, x)
    }
}

impl<const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<(RangeY, RangeX)> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, (x, y): (RangeY, RangeX)) -> Self::Output {
        self.matrix.slice(&(y, x))
    }
}

// TODO: &, &mut

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<[(usize, usize); N]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: [(usize, usize); N]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<&mut [(usize, usize); N]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &mut [(usize, usize); N]) -> Self::Output {
        self.slice(coords as &[(usize, usize); N])
    }
}

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<&[(usize, usize); N]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &[(usize, usize); N]) -> Self::Output {
        let mtx = self.matrix.data();
        let data = Box::new(coords.map(|(x, y)| mtx[x * X + y]));

        Vector::from(data)
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<[[(usize, usize); X2]; Y2]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&mut [[(usize, usize); X2]; Y2]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &mut [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(coords as &[[(usize, usize); X2]; Y2])
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&[[(usize, usize); X2]; Y2]> for IntoByColumn<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &[[(usize, usize); X2]; Y2]) -> Self::Output {
        let mtx = self.matrix.data();
        Matrix::from(coords.map(|row| row.map(|(x, y)| mtx[x * X + y])))
    }
}

// ======== ByColumn ===========================================================

#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByColumn<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a Matrix<X, Y, T>
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
    pub fn to_matrix(&self) -> Matrix<Y, X, T>
    where [T; Y * X]: Sized {
        self.matrix.transpose()
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> MatrixAggregate<Y, X, T> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    fn fold<F, B>(&self, init: B, f: F) -> Vector<X, B>
    where F: FnMut(B, T) -> B, B: Number {
        fold_by_column(self.matrix, init, f)
    }

    #[inline]
    fn mean(&self) -> Vector<X, T>
    where T: Ord, [T; X * Y]: Sized {
        mean_by_column(self.matrix)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Get<(usize, usize)> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn get(&self, (x, y): (usize, usize)) -> Option<&Self::Output> {
        self.matrix.get(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Index<(usize, usize)> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.matrix.index(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<(RangeY, RangeX)> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, (x, y): (RangeY, RangeX)) -> Self::Output {
        self.matrix.slice(&(y, x))
    }
}

// TODO: &, &mut

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<[(usize, usize); N]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: [(usize, usize); N]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<&mut [(usize, usize); N]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &mut [(usize, usize); N]) -> Self::Output {
        self.slice(coords as &[(usize, usize); N])
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<&[(usize, usize); N]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &[(usize, usize); N]) -> Self::Output {
        let mtx = self.matrix.data();
        let data = Box::new(coords.map(|(x, y)| mtx[x * X + y]));

        Vector::from(data)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<[[(usize, usize); X2]; Y2]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&mut [[(usize, usize); X2]; Y2]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &mut [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(coords as &[[(usize, usize); X2]; Y2])
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&[[(usize, usize); X2]; Y2]> for ByColumn<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &[[(usize, usize); X2]; Y2]) -> Self::Output {
        let mtx = self.matrix.data();
        Matrix::from(coords.map(|row| row.map(|(x, y)| mtx[x * X + y])))
    }
}

// ======== ByColumnMut ========================================================

#[repr(transparent)]
#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct ByColumnMut<'a, const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    pub matrix: &'a mut Matrix<X, Y, T>
}

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

    #[inline]
    pub fn to_matrix(&self) -> Matrix<Y, X, T>
    where [T; Y * X]: Sized {
        self.matrix.transpose()
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> MatrixAggregate<Y, X, T> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    fn fold<F, B>(&self, init: B, f: F) -> Vector<X, B>
    where F: FnMut(B, T) -> B, B: Number {
        fold_by_column(self.matrix, init, f)
    }

    #[inline]
    fn mean(&self) -> Vector<X, T>
    where T: Ord, [T; X * Y]: Sized {
        mean_by_column(self.matrix)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Get<(usize, usize)> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn get(&self, (x, y): (usize, usize)) -> Option<&Self::Output> {
        self.matrix.get(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> GetMut<(usize, usize)> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn get_mut(&mut self, (x, y): (usize, usize)) -> Option<&mut Self::Output> {
        self.matrix.get_mut(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Index<(usize, usize)> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.matrix.index(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> IndexMut<(usize, usize)> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.matrix.index_mut(y, x)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<(RangeY, RangeX)> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, (x, y): (RangeY, RangeX)) -> Self::Output {
        self.matrix.slice(&(y, x))
    }
}

// TODO: &, &mut

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<[(usize, usize); N]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: [(usize, usize); N]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<&mut [(usize, usize); N]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &mut [(usize, usize); N]) -> Self::Output {
        self.slice(coords as &[(usize, usize); N])
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const N: usize> Slice<&[(usize, usize); N]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &[(usize, usize); N]) -> Self::Output {
        let mtx = self.matrix.data();
        let data = Box::new(coords.map(|(x, y)| mtx[x * X + y]));

        Vector::from(data)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<[[(usize, usize); X2]; Y2]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&mut [[(usize, usize); X2]; Y2]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &mut [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(coords as &[[(usize, usize); X2]; Y2])
    }
}

impl<'a, const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&[[(usize, usize); X2]; Y2]> for ByColumnMut<'a, X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &[[(usize, usize); X2]; Y2]) -> Self::Output {
        let mtx = self.matrix.data();
        Matrix::from(coords.map(|row| row.map(|(x, y)| mtx[x * X + y])))
    }
}

// ======== Helper Functions ===================================================

#[inline]
pub fn mean_by_column<const X: usize, const Y: usize, T: Number>(matrix: &Matrix<X, Y, T>) -> Vector<X, T>
where [T; X * Y]: Sized, T: Ord {
    let mut iter = matrix.columns();
    let data = Box::new([(); X].map(|_| {
        let mut vector = iter.next().unwrap();
        vector.sort();
        if Y & 1 != 0 {
            vector[Y / 2]
        } else {
            let index = Y / 2;
            (vector[index - 1] + vector[index]) / (T::ONE + T::ONE)
        }
    }));

    Vector::from(data)
}

#[inline]
pub fn fold_by_column<F, B, const X: usize, const Y: usize, T: Number>(matrix: &Matrix<X, Y, T>, init: B, mut f: F) -> Vector<X, B>
where F: FnMut(B, T) -> B, B: Number, [T; X * Y]: Sized {
    let mtx = matrix.data();
    let mut data = Box::new([init; X]);

    for y in 0..Y {
        let yoffset = y * X;
        for (value, res) in mtx[yoffset..yoffset + X].iter().zip(data.iter_mut()) {
            *res = f(*res, *value);
        }
    }

    Vector::from(data)
}
