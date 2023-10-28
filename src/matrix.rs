use std::mem::replace;
use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign, Index, IndexMut};
use std::iter::{Sum, Product, IntoIterator};
use std::fmt::{Display, Debug};

use crate::Vector;
use crate::assert::{Assert, IsTrue};
use crate::bycolumn::{ByColumn, ByColumnMut, IntoByColumn};
use crate::number::Number;
use crate::ops::{Get, GetMut, Pow, PowAssign, Unit, Dot, DotAssign, Slice};
use crate::range::{RangeIter, Range};

#[repr(transparent)]
#[derive(Clone)]
pub struct Matrix<const X: usize, const Y: usize, T: Number=f64>
where [T; X * Y]: Sized
{
    data: Box<[T; X * Y]>
}

impl<const X: usize, const Y: usize, T: Number> Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    pub const X: usize = X;
    pub const Y: usize = Y;
    pub const SHAPE: [usize; 2] = [Y, X];

    #[inline]
    pub fn iter(&self) -> impl std::iter::Iterator<Item = &T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl std::iter::Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn iter_arrays(&self) -> impl std::iter::Iterator<Item = &[T; X]> {
        unsafe { self.data.as_chunks_unchecked::<X>() }.iter()
    }

    #[inline]
    pub fn iter_arrays_mut(&mut self) -> impl std::iter::Iterator<Item = &mut [T; X]> {
        unsafe { self.data.as_chunks_unchecked_mut::<X>() }.iter_mut()
    }

    #[inline]
    pub fn iter_vectors(&self) -> impl std::iter::Iterator<Item = Vector<X, T>> + '_ {
        unsafe { self.data.as_chunks_unchecked::<X>() }.iter().map(Vector::from)
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize) -> Option<&T> {
        self.data.get(y * X + x)
    }

    #[inline]
    pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
        self.data.get_mut(y * X + x)
    }

    #[inline]
    fn index(&self, x: usize, y: usize) -> &T {
        &self.data[X * y + x]
    }

    #[inline]
    fn index_mut(&mut self, x: usize, y: usize) -> &mut T {
        &mut self.data[X * y + x]
    }

    #[inline]
    pub fn data(&self) -> &[T; X * Y] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T; X * Y] {
        &mut self.data
    }

    #[inline]
    pub fn into_data(self) -> [T; X * Y] {
        *self.data
    }

    #[inline]
    pub fn into_vector(self) -> Vector<{ X * Y }, T> {
        self.data.into()
    }

    #[inline]
    pub fn reshape<const X2: usize, const Y2: usize>(&self) -> Matrix<X2, Y2, T>
    where [T; X2 * Y2]: Sized, Assert<{ X * Y == X2 * Y2 }>: IsTrue {
        Matrix::from(self)
    }

    #[inline]
    pub fn into_reshape<const X2: usize, const Y2: usize>(self) -> Matrix<X2, Y2, T>
    where [T; X2 * Y2]: Sized, Assert<{ X * Y == X2 * Y2 }>: IsTrue {
        // XXX: is this correct?
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Matrix<X, Y, U>
    where F: FnMut(T) -> U, U: Number {
        Matrix { data: Box::new(self.data.map(f)) }
    }

    #[inline]
    pub fn map_assign<F, U>(&mut self, mut f: F)
    where F: FnMut(T) -> T {
        for x in self.data.iter_mut() {
            *x = f(*x);
        }
    }

    pub fn transpose(&self) -> Matrix<Y, X, T>
    where [T; Y * X]: Sized {
        // TODO: MaybeUninit?
        let mut data = Box::new([T::default(); Y * X]);

        for y in 0..Y {
            let yoffset = y * X;
            for x in 0..X {
                data[x * Y + y] = self.data[yoffset + x];
            }
        }

        Matrix { data }
    }

    pub fn transpose_map<F, U>(&self, mut f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        // TODO: MaybeUninit?
        let mut data = Box::new([U::default(); Y * X]);

        for y in 0..Y {
            let yoffset = y * X;
            for x in 0..X {
                data[x * Y + y] = f(self.data[yoffset + x]);
            }
        }

        Matrix::from(data)
    }

    #[inline]
    pub fn transpose_assign(&mut self)
    where [T; Y * X]: Sized, Assert<{ X == Y }>: IsTrue {
        for y in 1..Y {
            let yoffset = y * X;
            for x in 0..y {
                self.data.swap(yoffset + x, x * X + y);
            }
        }
    }

    #[inline]
    pub fn into_by_column(self) -> IntoByColumn<X, Y, T> {
        IntoByColumn::new(self)
    }

    #[inline]
    pub fn by_column(&self) -> ByColumn<'_, X, Y, T> {
        ByColumn::new(self)
    }

    #[inline]
    pub fn by_column_mut(&mut self) -> ByColumnMut<'_, X, Y, T> {
        ByColumnMut::new(self)
    }

    #[inline]
    pub const fn range_x(&self) -> Range::<0, X> {
        Range::<0, X>()
    }

    #[inline]
    pub const fn range_y(&self) -> Range::<0, Y> {
        Range::<0, Y>()
    }

    #[inline]
    pub const fn range_xy(&self) -> (Range::<0, X>, Range::<0, Y>) {
        (Range::<0, X>(), Range::<0, Y>())
    }
}

impl<const X: usize, const Y: usize, T: Number> IntoIterator for Matrix<X, Y, T>
where [T; X * Y]: Sized {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, { X * Y }>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<const X: usize, const Y: usize, T: Number> AsRef<Self> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<const X: usize, const Y: usize, T: Number> AsRef<[[T; X]]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn as_ref(&self) -> &[[T; X]] {
        unsafe { self.data.as_chunks_unchecked::<X>() }
    }
}

impl<const N: usize, T: Number> Unit for Matrix<N, N, T>
where [T; N * N]: Sized {
    #[inline]
    fn unit() -> Self {
        let mut data = Box::new([T::default(); N * N]);

        for index in 0..N {
            data[N * index + index] = T::ONE;
        }

        Self { data }
    }
}

impl<const X: usize, const Y: usize, T: Number> Default for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn default() -> Self {
        Self {
            data: Box::new([Default::default(); X * Y])
        }
    }
}

impl<const X: usize, const Y: usize, T: Number> Display for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let table = self.data.map(|value| {
            let cell = format!("{:?}", value);
            let (prefix, suffix) = get_prefix_suffix(&cell);
            (cell, prefix, suffix)
        });
        let mut widths = [(0usize, 0usize); X];

        fn get_prefix_suffix(cell: &str) -> (usize, usize) {
            let len = cell.len();
            let prefix;
            let suffix;
            if let Some(index) = cell.find('.') {
                prefix = index;
                suffix = len - 1 - index;
            } else {
                prefix = len;
                suffix = 0;
            }

            (prefix, suffix)
        }

        for y in 0..Y {
            let yoffset = y * X;
            for x in 0..X {
                let (_, prefix, suffix) = table[yoffset + x];

                if prefix > widths[x].0 {
                    widths[x].0 = prefix;
                }

                if suffix > widths[x].1 {
                    widths[x].1 = suffix;
                }
            }
        }

        Display::fmt("[\n", f)?;

        for y in 0..Y {
            let yoffset = y * X;
            Display::fmt("    [", f)?;
            for x in 0..X {
                let (col_prefix, col_suffix) = widths[x];
                let (ref cell, prefix, suffix) = table[yoffset + x];

                for _ in 0..col_prefix - prefix + usize::from(x > 0) * 2 {
                    Display::fmt(&' ', f)?;
                }

                Display::fmt(cell, f)?;

                for _ in 0..col_suffix - suffix {
                    Display::fmt(&' ', f)?;
                }
            }
            Display::fmt("]\n", f)?;
        }

        Display::fmt(&']', f)
    }
}

impl<const X: usize, const Y: usize, T: Number> Debug for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt("Matrix::from(", f)?;
        unsafe { self.data.as_chunks_unchecked::<X>() }.fmt(f)?;
        Display::fmt(&')', f)
    }
}

// ======== Equality ===========================================================

impl<const X: usize, const Y: usize, T: Number> PartialEq for Matrix<X, Y, T>
where T: PartialEq, [T; X * Y]: Sized {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<const X: usize, const Y: usize, T: Number> PartialEq<[[T; X]; Y]> for Matrix<X, Y, T>
where T: PartialEq, [T; X * Y]: Sized {
    #[inline]
    fn eq(&self, other: &[[T; X]; Y]) -> bool {
        let other: &[T; X * Y] = unsafe { std::mem::transmute(other) };
        &*self.data == other
    }
}

impl<const X: usize, const Y: usize, T: Number> PartialEq<Matrix<X, Y, T>> for [[T; X]; Y]
where T: PartialEq, [T; X * Y]: Sized {
    #[inline]
    fn eq(&self, other: &Matrix<X, Y, T>) -> bool {
        let data: &[T; X * Y] = unsafe { std::mem::transmute(self) };
        data == &*other.data
    }
}

impl<const X: usize, const Y: usize, T: Number> PartialEq<[&[T; X]; Y]> for Matrix<X, Y, T>
where T: PartialEq, [T; X * Y]: Sized {
    #[inline]
    fn eq(&self, other: &[&[T; X]; Y]) -> bool {
        self.iter_arrays().zip(other.iter()).all(|(a, b)| a == *b)
    }
}

impl<const X: usize, const Y: usize, T: Number> PartialEq<Matrix<X, Y, T>> for [&[T; X]; Y]
where T: PartialEq, [T; X * Y]: Sized {
    #[inline]
    fn eq(&self, other: &Matrix<X, Y, T>) -> bool {
        other == self
    }
}

impl<const X: usize, const Y: usize, T: Number> Eq for Matrix<X, Y, T>
where T: Eq, [T; X * Y]: Sized {}

// ======== From ===============================================================

impl<const X: usize, const Y: usize, T: Number> From<Box<[T; X * Y]>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Box<[T; X * Y]>) -> Self {
        Self { data: value }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<[T; X * Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: [T; X * Y]) -> Self {
        Self { data: Box::new(value) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<[[T; X]; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    // XXX: is this correct?
    #[inline]
    fn from(value: [[T; X]; Y]) -> Self {
        Self { data: Box::new(unsafe { std::mem::transmute_copy(&value) }) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Box<[[T; X]; Y]>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    // XXX: is this correct?
    #[inline]
    fn from(value: Box<[[T; X]; Y]>) -> Self {
        Self { data: unsafe { std::mem::transmute(value) } }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&[[T; X]; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    // XXX: is this correct?
    #[inline]
    fn from(value: &[[T; X]; Y]) -> Self {
        Self { data: Box::new(unsafe { std::mem::transmute_copy(value) }) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&[&[T; X]; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &[&[T; X]; Y]) -> Self {
        // TODO: optimize using MaybeUninit?
        let mut data = Box::new([Default::default(); X * Y]);
        for y in 0..Y {
            data[y * X..(y + 1) * X].copy_from_slice(value[y]);
        }
        Self { data }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<[&[T; X]; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: [&[T; X]; Y]) -> Self {
        Matrix::from(&value)
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&[Vector<X, T>; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &[Vector<X, T>; Y]) -> Self {
        // TODO: optimize using MaybeUninit?
        let mut data = Box::new([Default::default(); X * Y]);
        for y in 0..Y {
            data[y * X..(y + 1) * X].copy_from_slice(value[y].data());
        }
        Self { data }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<[Vector<X, T>; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: [Vector<X, T>; Y]) -> Self {
        Matrix::from(&value)
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&[&Vector<X, T>; Y]> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &[&Vector<X, T>; Y]) -> Self {
        // TODO: optimize using MaybeUninit?
        let mut data = Box::new([Default::default(); X * Y]);
        for y in 0..Y {
            data[y * X..(y + 1) * X].copy_from_slice(value[y].data());
        }
        Self { data }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Vector<{X * Y}, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Vector<{X * Y}, T>) -> Self {
        // XXX: is this correct?
        unsafe { std::mem::transmute(value) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&Vector<{X * Y}, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &Vector<{X * Y}, T>) -> Self {
        Self { data: Box::new(*value.data()) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<T> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: T) -> Self {
        Self { data: Box::new([value; X * Y]) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&T> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &T) -> Self {
        let value = *value;
        Self { data: Box::new([value; X * Y]) }
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Matrix<X, Y, T>> for [T; X * Y]
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Matrix<X, Y, T>) -> Self {
        *value.data
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Matrix<X, Y, T>> for Vector<{X * Y}, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Matrix<X, Y, T>) -> Self {
        Vector::from(value.data)
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&Matrix<X, Y, T>> for Vector<{X * Y}, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &Matrix<X, Y, T>) -> Self {
        Vector::from(value.data.clone())
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Matrix<X, Y, T>> for Box<[T; X * Y]>
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Matrix<X, Y, T>) -> Self {
        value.data
    }
}

impl<const X: usize, const Y: usize, T: Number> From<Matrix<X, Y, T>> for [[T; X]; Y]
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: Matrix<X, Y, T>) -> Self {
        <[[T; X]; Y]>::from(&value)
    }
}

impl<const X: usize, const Y: usize, T: Number> From<&Matrix<X, Y, T>> for [[T; X]; Y]
where [T; X * Y]: Sized
{
    #[inline]
    fn from(value: &Matrix<X, Y, T>) -> Self {
        let data = &*value.data;
        unsafe { std::mem::transmute_copy(data) }
    }
}

impl<const X1: usize, const Y1: usize, const X2: usize, const Y2: usize, T: Number> From<&Matrix<X1, Y1, T>> for Matrix<X2, Y2, T>
where [T; X1 * Y1]: Sized, [T; X2 * Y2]: Sized, Assert<{ X1 * Y1 == X2 * Y2 }>: IsTrue
{
    #[inline]
    fn from(value: &Matrix<X1, Y1, T>) -> Self {
        let data = &*value.data;
        Self { data: Box::new(unsafe { std::mem::transmute_copy(data) }) }
    }
}

// ======== Index ==============================================================

impl<const X: usize, const Y: usize, T: Number> Index<usize> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = [T; X];

    #[inline]
    fn index(&self, y: usize) -> &Self::Output {
        unsafe { self.data.as_chunks_unchecked::<X>() }.index(y)
    }
}

impl<const X: usize, const Y: usize, T: Number> Index<(usize, usize)> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn index(&self, (x, y): (usize, usize)) -> &Self::Output {
        self.index(x, y)
    }
}

impl<const X: usize, const Y: usize, T: Number> IndexMut<usize> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn index_mut(&mut self, y: usize) -> &mut Self::Output {
        unsafe { self.data.as_chunks_unchecked_mut::<X>() }.index_mut(y)
    }
}

impl<const X: usize, const Y: usize, T: Number> IndexMut<(usize, usize)> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn index_mut(&mut self, (x, y): (usize, usize)) -> &mut Self::Output {
        self.index_mut(x, y)
    }
}

// ======== Get ================================================================

impl<const X: usize, const Y: usize, T: Number> Get<usize> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = [T; X];

    #[inline]
    fn get(&self, y: usize) -> Option<&Self::Output> {
        unsafe { self.data.as_chunks_unchecked::<X>() }.get(y)
    }
}

impl<const X: usize, const Y: usize, T: Number> Get<(usize, usize)> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = T;

    #[inline]
    fn get(&self, (x, y): (usize, usize)) -> Option<&Self::Output> {
        self.data.get(X * y + x)
    }
}

impl<const X: usize, const Y: usize, T: Number> GetMut<usize> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn get_mut(&mut self, y: usize) -> Option<&mut Self::Output> {
        unsafe { self.data.as_chunks_unchecked_mut::<X>() }.get_mut(y)
    }
}

impl<const X: usize, const Y: usize, T: Number> GetMut<(usize, usize)> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn get_mut(&mut self, (x, y): (usize, usize)) -> Option<&mut Self::Output> {
        self.data.get_mut(X * y + x)
    }
}

// ======== Slice ==============================================================

impl<const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<(RangeX, RangeY)> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, ranges: (RangeX, RangeY)) -> Self::Output {
        self.slice(&ranges)
    }
}

impl<const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<&mut (RangeX, RangeY)> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, ranges: &mut (RangeX, RangeY)) -> Self::Output {
        self.slice(ranges as &(RangeX, RangeY))
    }
}

impl<const X: usize, const Y: usize, T: Number, RangeX: RangeIter, RangeY: RangeIter> Slice<&(RangeX, RangeY)> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; RangeX::LEN * RangeY::LEN]: Sized
{
    type Output = Matrix<{ RangeX::LEN }, { RangeY::LEN }, T>;

    #[inline]
    fn slice(&self, (x_range, y_range): &(RangeX, RangeY)) -> Self::Output {
        let mut data = Box::new([T::default(); RangeX::LEN * RangeY::LEN]);

        let x_iter = x_range.iter();
        let mut index = 0;
        for y in y_range.iter() {
            let yoffset = y * X;
            for x in x_iter.clone() {
                data[index] = self.data[yoffset + x];
                index += 1;
            }
        }

        Matrix { data }
    }
}

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<[(usize, usize); N]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: [(usize, usize); N]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<&mut [(usize, usize); N]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &mut [(usize, usize); N]) -> Self::Output {
        self.slice(coords as &[(usize, usize); N])
    }
}

impl<const X: usize, const Y: usize, T: Number, const N: usize> Slice<&[(usize, usize); N]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn slice(&self, coords: &[(usize, usize); N]) -> Self::Output {
        let data = Box::new(coords.map(|(x, y)| self.data[y * X + x]));

        Vector::from(data)
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<[[(usize, usize); X2]; Y2]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(&coords)
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&mut [[(usize, usize); X2]; Y2]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &mut [[(usize, usize); X2]; Y2]) -> Self::Output {
        self.slice(coords as &[[(usize, usize); X2]; Y2])
    }
}

impl<const X: usize, const Y: usize, T: Number, const X2: usize, const Y2: usize> Slice<&[[(usize, usize); X2]; Y2]> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; X2 * Y2]: Sized
{
    type Output = Matrix<X2, Y2, T>;

    #[inline]
    fn slice(&self, coords: &[[(usize, usize); X2]; Y2]) -> Self::Output {
        Matrix::from(coords.map(|row| row.map(|(x, y)| self.data[y * X + x])))
    }
}

// ======== Arithmetic Operations ==============================================

macro_rules! impl_ops {
    (@move_rhs @commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, T>) -> Self::Output {
                rhs.$op_assign(self);

                rhs
            }
        }
    };

    (@move_rhs @non_commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, T>) -> Self::Output {
                for index in 0..X * Y {
                    rhs.data[index] = self.data[index].$op(rhs.data[index]);
                }

                rhs
            }
        }
    };

    (@primitive $type:ident @$commutative:ident Pow PowAssign pow pow_assign $(where $($where:tt)*)?) => {
        // more limitations, see below
    };

    (@primitive $type:ident @commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const X: usize, const Y: usize> $trait<Matrix<X, Y, $type>> for $type
        where [$type; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, $type>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, $type>) -> Self::Output {
                rhs.$op_assign(self);

                rhs
            }
        }
    };

    (@primitive $type:ident @non_commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const X: usize, const Y: usize> $trait<Matrix<X, Y, $type>> for $type
        where [$type; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, $type>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, $type>) -> Self::Output {
                for x in rhs.data.iter_mut() {
                    *x = self.$op(*x);
                }

                rhs
            }
        }
    };

    (@$commutative:ident $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl_ops!(@primitive i8    @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive i16   @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive i32   @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive i64   @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive i128  @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive isize @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive f32   @$commutative $trait $trait_assign $op $op_assign);
        impl_ops!(@primitive f64   @$commutative $trait $trait_assign $op $op_assign);

        // ======== move =======================================================

        impl<const X: usize, const Y: usize, T: Number> $trait for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: &Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<T> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: T) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: Vector<X, T>) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: &Vector<X, T>) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for IntoByColumn<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<Y, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for IntoByColumn<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(mut self, rhs: &Vector<Y, T>) -> Self::Output {
                by_column::matrix_vector::$op(&mut self.matrix, rhs.data());

                self.matrix
            }
        }

        // ======== ref ========================================================

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);

                res
            }
        }

        impl_ops!(@move_rhs @$commutative $trait $trait_assign $op $op_assign $(where $($where)*)?);

        impl<const X: usize, const Y: usize, T: Number> $trait<T> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<X, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<X, T>) -> Self::Output {
                let mut res = self.clone();
                by_row::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&mut Vector<X, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &mut Vector<X, T>) -> Self::Output {
                let mut res = self.clone();
                by_row::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&[T; X]> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &[T; X]) -> Self::Output {
                let mut res = self.clone();
                by_row::matrix_vector::$op(&mut res, rhs);

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&mut [T; X]> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &mut [T; X]) -> Self::Output {
                let mut res = self.clone();
                by_row::matrix_vector::$op(&mut res, rhs);

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<[T; X]> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: [T; X]) -> Self::Output {
                let mut res = self.clone();
                by_row::matrix_vector::$op(&mut res, &rhs);

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for ByColumn<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<Y, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                by_column::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for ByColumn<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<Y, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                by_column::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&mut Vector<Y, T>> for ByColumn<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &mut Vector<Y, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                by_column::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        // ======== mut ref ====================================================

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<T> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<X, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<X, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for ByColumnMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<Y, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for ByColumnMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<Y, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                by_column::matrix_vector::$op(&mut res, rhs.data());

                res
            }
        }

        // ======== assign =====================================================

        impl<const X: usize, const Y: usize, T: Number> $trait_assign for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Self) {
                self.$op_assign(&rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&Matrix<X, Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Self) {
                for index in 0..X * Y {
                    self.data[index].$op_assign(rhs.data[index]);
                }
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<T> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: T) {
                for index in 0..X * Y {
                    self.data[index].$op_assign(rhs);
                }
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&T> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &T) {
                let rhs = *rhs;
                for index in 0..X * Y {
                    self.data[index].$op_assign(rhs);
                }
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<Vector<X, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Vector<X, T>) {
                self.$op_assign(&rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&Vector<X, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Vector<X, T>) {
                by_row::matrix_vector::$op(self, rhs.data());
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&[T; X]> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &[T; X]) {
                by_row::matrix_vector::$op(self, rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&mut [T; X]> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &mut [T; X]) {
                by_row::matrix_vector::$op(self, rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<[T; X]> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: [T; X]) {
                by_row::matrix_vector::$op(self, &rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<Vector<Y, T>> for ByColumnMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Vector<Y, T>) {
                self.$op_assign(&rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&Vector<Y, T>> for ByColumnMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Vector<Y, T>) {
                by_column::matrix_vector::$op(&mut self.matrix, rhs.data());
            }
        }

        // ======== Vector x Matrix ============================================

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, T>) -> Self::Output {
                by_row::vector_matrix::$op(self.data(), &mut rhs);

                rhs
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                self.$op(rhs.clone())
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &mut Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                (self as &Vector<X, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &mut Vector<X, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                (self as &Vector<X, T>).$op(rhs)
            }
        }

        // ======== ByColumn x Vector ==========================================

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumn<'_, X, Y, T>> for Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumn<'_, X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumn<'_, X, Y, T>> for &mut Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumn<'_, X, Y, T>) -> Self::Output {
                (self as &Vector<Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumn<'_, X, Y, T>> for &Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumn<'_, X, Y, T>) -> Self::Output {
                let mut res = rhs.matrix.clone();
                by_column::vector_matrix::$op(self.data(), &mut res);

                res
            }
        }

        // ======== ByColumnMut x Vector =======================================

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumnMut<'_, X, Y, T>> for Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumnMut<'_, X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumnMut<'_, X, Y, T>> for &mut Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumnMut<'_, X, Y, T>) -> Self::Output {
                (self as &Vector<Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<ByColumnMut<'_, X, Y, T>> for &Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: ByColumnMut<'_, X, Y, T>) -> Self::Output {
                let mut res = rhs.matrix.clone();
                by_column::vector_matrix::$op(self.data(), &mut res);

                res
            }
        }

        // ======== IntoByColumn x Vector ======================================

        impl<const X: usize, const Y: usize, T: Number> $trait<IntoByColumn<X, Y, T>> for Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: IntoByColumn<X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<IntoByColumn<X, Y, T>> for &mut Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: IntoByColumn<X, Y, T>) -> Self::Output {
                (self as &Vector<Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<IntoByColumn<X, Y, T>> for &Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, mut rhs: IntoByColumn<X, Y, T>) -> Self::Output {
                by_column::vector_matrix::$op(self.data(), &mut rhs.matrix);

                rhs.matrix
            }
        }
    };
}

impl_ops!(@commutative     Add AddAssign add add_assign);
impl_ops!(@non_commutative Sub SubAssign sub sub_assign);
impl_ops!(@commutative     Mul MulAssign mul mul_assign);
impl_ops!(@non_commutative Div DivAssign div div_assign);
impl_ops!(@non_commutative Pow PowAssign pow pow_assign where T: PowAssign + Pow<Output = T>);

macro_rules! impl_helper {
    (@$dir:ident @$sides:ident [$(,)?]) => {};

    (@by_row @vector_matrix [[$op:ident $op_assign:ident $(where $($where:tt)*)?] $($rest:tt)*]) => {
        pub fn $op<const X: usize, const Y: usize, T: crate::Number>(
            lhs: &[T; X],
            rhs: &mut crate::Matrix<X, Y, T>
        )
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            let rhs = &mut rhs.data;
            for y in 0..Y {
                let yoffset = X * y;
                for x in 0..X {
                    let index = yoffset + x;
                    rhs[index] = lhs[x].$op(rhs[index]);
                }
            }
        }

        impl_helper!(@by_row @vector_matrix [$($rest)*]);
    };

    (@by_row @matrix_vector [[$op:ident $op_assign:ident $(where $($where:tt)*)?] $($rest:tt)*]) => {
        pub fn $op<const X: usize, const Y: usize, T: crate::Number>(
            lhs: &mut crate::Matrix<X, Y, T>,
            rhs: &[T; X]
        )
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            let lhs = &mut lhs.data;
            for y in 0..Y {
                let yoffset = X * y;
                for x in 0..X {
                    lhs[yoffset + x].$op_assign(rhs[x]);
                }
            }
        }

        impl_helper!(@by_row @matrix_vector [$($rest)*]);
    };

    (@by_column @vector_matrix [[$op:ident $op_assign:ident $(where $($where:tt)*)?] $($rest:tt)*]) => {
        pub fn $op<const X: usize, const Y: usize, T: crate::Number>(
            lhs: &[T; Y],
            rhs: &mut crate::Matrix<X, Y, T>
        )
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            let rhs = &mut rhs.data;
            for y in 0..Y {
                let yoffset = X * y;
                let lhs_y = lhs[y];
                for x in 0..X {
                    let index = yoffset + x;
                    rhs[index] = lhs_y.$op(rhs[index]);
                }
            }
        }

        impl_helper!(@by_column @vector_matrix [$($rest)*]);
    };

    (@by_column @matrix_vector [[$op:ident $op_assign:ident $(where $($where:tt)*)?] $($rest:tt)*]) => {
        pub fn $op<const X: usize, const Y: usize, T: crate::Number>(
            lhs: &mut crate::Matrix<X, Y, T>,
            rhs: &[T; Y]
        )
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            let lhs = &mut lhs.data;
            for y in 0..Y {
                let yoffset = X * y;
                let rhs_y = rhs[y];
                for x in 0..X {
                    lhs[yoffset + x].$op_assign(rhs_y);
                }
            }
        }

        impl_helper!(@by_column @matrix_vector [$($rest)*]);
    };

    ($($args:tt)*) => {
        pub mod by_row {
            pub mod vector_matrix {
                impl_helper!(@by_row @vector_matrix [$($args)*]);
            }

            pub mod matrix_vector {
                impl_helper!(@by_row @matrix_vector [$($args)*]);
            }
        }

        pub mod by_column {
            pub mod vector_matrix {
                impl_helper!(@by_column @vector_matrix [$($args)*]);
            }

            pub mod matrix_vector {
                impl_helper!(@by_column @matrix_vector [$($args)*]);
            }
        }
    };
}

impl_helper! {
    [add add_assign]
    [sub sub_assign]
    [mul mul_assign]
    [div div_assign]
    [pow pow_assign where T: crate::ops::PowAssign + crate::ops::Pow<Output = T>]
}

// ======== Pow ================================================================

macro_rules! impl_pow {
    ($type:ident $($cast:tt)*) => {
        impl<const X: usize, const Y: usize> Pow<Matrix<X, Y, Self>> for $type
        where [Self; X * Y]: Sized
        {
            type Output = Matrix<X, Y, Self>;

            #[inline]
            fn pow(self, mut rhs: Matrix<X, Y, Self>) -> Self::Output {
                for x in rhs.data.iter_mut() {
                    *x = self.pow(*x $($cast)*);
                }

                rhs
            }
        }
    };
}

impl_pow!(i8 as u32);
impl_pow!(i16 as u32);
impl_pow!(f32);
impl_pow!(f64);

impl<const X: usize, const Y: usize> Pow<&Matrix<X, Y, Self>> for f32
where [Self; X * Y]: Sized
{
    type Output = Matrix<X, Y, Self>;

    #[inline]
    fn pow(self, rhs: &Matrix<X, Y, Self>) -> Self::Output {
        let mut res: Matrix<X, Y, Self> = self.into();
        res.pow_assign(rhs);

        res
    }
}

impl<const X: usize, const Y: usize> Pow<&Matrix<X, Y, Self>> for f64
where [Self; X * Y]: Sized
{
    type Output = Matrix<X, Y, Self>;

    #[inline]
    fn pow(self, rhs: &Matrix<X, Y, Self>) -> Self::Output {
        let mut res: Matrix<X, Y, Self> = self.into();
        res.pow_assign(rhs);

        res
    }
}

// ======== Neg ================================================================

impl<const X: usize, const Y: usize, T: Number> Neg for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = Self;

    #[inline]
    fn neg(mut self) -> Self::Output {
        for x in self.data.iter_mut() {
            *x = -*x;
        }
        self
    }
}

impl<const X: usize, const Y: usize, T: Number> Neg for &Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = Matrix<X, Y, T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Matrix {
            data: Box::new(self.data.map(|value| -value))
        }
    }
}

impl<const X: usize, const Y: usize, T: Number> Neg for &mut Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    type Output = Matrix<X, Y, T>;

    #[inline]
    fn neg(self) -> Self::Output {
        (self as &Matrix<X, Y, T>).neg()
    }
}

// ======== Sum ================================================================

impl<const X: usize, const Y: usize, T: Number> Sum for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(mut res) = iter.next() {
            for item in iter {
                res += item;
            }
            return res;
        }
        Matrix::default()
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Sum<&'a Matrix<X, Y, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            let mut res = first.clone();
            for item in iter {
                res += item;
            }
            return res;
        }
        Matrix::default()
    }
}

// ======== Product ============================================================

impl<const X: usize, const Y: usize, T: Number> Product for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    fn product<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(mut res) = iter.next() {
            for item in iter {
                res *= item;
            }
            return res;
        }
        T::ONE.into()
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> Product<&'a Matrix<X, Y, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    fn product<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            let mut res = first.clone();
            for item in iter {
                res *= item;
            }
            return res;
        }
        T::ONE.into()
    }
}

// ======== Dot ================================================================

impl<const X: usize, const Y: usize, T: Number> Dot<Matrix<Y, X, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    #[inline]
    fn dot(self, rhs: Matrix<Y, X, T>) -> Self::Output {
        (&self).dot(&rhs)
    }
}

impl<const X: usize, const Y: usize, T: Number> Dot<&Matrix<Y, X, T>> for Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    #[inline]
    fn dot(self, rhs: &Matrix<Y, X, T>) -> Self::Output {
        (&self).dot(rhs)
    }
}

impl<const X: usize, const Y: usize, T: Number> Dot<Matrix<Y, X, T>> for &Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    #[inline]
    fn dot(self, rhs: Matrix<Y, X, T>) -> Self::Output {
        self.dot(&rhs)
    }
}

impl<const X: usize, const Y: usize, T: Number> Dot<Matrix<Y, X, T>> for &mut Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    #[inline]
    fn dot(self, rhs: Matrix<Y, X, T>) -> Self::Output {
        (self as &Matrix<X, Y, T>).dot(&rhs)
    }
}

impl<const X: usize, const Y: usize, T: Number> Dot<&Matrix<Y, X, T>> for &mut Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    #[inline]
    fn dot(self, rhs: &Matrix<Y, X, T>) -> Self::Output {
        (self as &Matrix<X, Y, T>).dot(rhs)
    }
}

impl<const X: usize, const Y: usize, T: Number> Dot<&Matrix<Y, X, T>> for &Matrix<X, Y, T>
where [T; X * Y]: Sized, [T; Y * X]: Sized, [T; Y * Y]: Sized {
    type Output = Matrix<Y, Y, T>;

    fn dot(self, rhs: &Matrix<Y, X, T>) -> Self::Output {
        let mut data = Box::new([T::default(); Y * Y]);

        for y in 0..Y {
            let lhs_yoffset = X * y;
            let res_yoffset = Y * y;
            for x in 0..Y {
                let mut value = T::ZERO;
                for z in 0..X {
                    value += self.data[lhs_yoffset + z] * rhs.data[z * Y + x];
                }
                data[res_yoffset + x] = value;
            }
        }

        Matrix { data }
    }
}

impl<const N: usize, T: Number> DotAssign<&Matrix<N, N, T>> for Matrix<N, N, T>
where [T; N * N]: Sized {
    #[inline]
    fn dot_assign(&mut self, rhs: &Matrix<N, N, T>) {
        let res = self.dot(rhs);
        let _ = replace(&mut self.data, res.data);
    }
}

// TODO: impl Cross, CrossAssign, EigenValue, EigenVector etc.
