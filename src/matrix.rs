use std::mem::replace;
use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign, Index, IndexMut};
use std::iter::{Sum, Product, IntoIterator};
use std::fmt::{Display, Debug};

use crate::Vector;
use crate::assert::{Assert, IsTrue};
use crate::byrow::{ByRow, ByRowMut, IntoByRow};
use crate::number::Number;
use crate::ops::{Get, GetMut, Pow, PowAssign, Unit, Dot, DotAssign};

#[repr(transparent)]
#[derive(PartialEq, Clone)]
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
        unsafe { self.data.as_chunks_unchecked::<X>() }.into_iter()
    }

    #[inline]
    pub fn iter_arrays_mut(&mut self) -> impl std::iter::Iterator<Item = &mut [T; X]> {
        unsafe { self.data.as_chunks_unchecked_mut::<X>() }.into_iter()
    }

    #[inline]
    pub fn iter_vectors<'a>(&'a self) -> impl std::iter::Iterator<Item = Vector<X, T>> + 'a {
        unsafe { self.data.as_chunks_unchecked::<X>() }.into_iter().map(Vector::from)
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
    pub fn into_by_row(self) -> IntoByRow<X, Y, T> {
        IntoByRow::new(self)
    }

    #[inline]
    pub fn by_row(&self) -> ByRow<'_, X, Y, T> {
        ByRow::new(self)
    }

    #[inline]
    pub fn by_row_mut(&mut self) -> ByRowMut<'_, X, Y, T> {
        ByRowMut::new(self)
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
        unsafe { self.data.as_chunks_unchecked::<X>() }.fmt(f)
    }
}

impl<const X: usize, const Y: usize, T: Number> Debug for Matrix<X, Y, T>
where [T; X * Y]: Sized
{
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unsafe { self.data.as_chunks_unchecked::<X>() }.fmt(f)
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
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(&rhs);

                res
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
            fn $op(self, rhs: Matrix<X, Y, $type>) -> Self::Output {
                let mut res: Self::Output = self.into();
                res.$op_assign(&rhs);

                res
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
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: &Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<T> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: T) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: Vector<Y, T>) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: &Vector<Y, T>) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for IntoByRow<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(self, rhs: Vector<X, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for IntoByRow<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: &Vector<X, T>) -> Self::Output {
                for y in 0..Y {
                    let yoffset = X * y;
                    for x in 0..X {
                        self.matrix.data[yoffset + x].$op_assign(rhs[x]);
                    }
                }
                self
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

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<Y, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for &Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<Y, T>) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);

                res
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for ByRow<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<X, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for ByRow<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<X, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                for y in 0..Y {
                    let yoffset = X * y;
                    for x in 0..X {
                        res.data[yoffset + x].$op_assign(rhs[x]);
                    }
                }

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

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<Y, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<Y, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<Y, T>> for &mut Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<Y, T>) -> Self::Output {
                (self as &Matrix<X, Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Vector<X, T>> for ByRowMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Vector<X, T>) -> Self::Output {
                self.$op(&rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Vector<X, T>> for ByRowMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Vector<X, T>) -> Self::Output {
                let mut res = self.matrix.clone();
                for y in 0..Y {
                    let yoffset = X * y;
                    for x in 0..X {
                        res.data[yoffset + x].$op_assign(rhs[x]);
                    }
                }

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

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<Vector<Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Vector<Y, T>) {
                self.$op_assign(&rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&Vector<Y, T>> for Matrix<X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Vector<Y, T>) {
                for y in 0..Y {
                    let yoffset = X * y;
                    for x in 0..X {
                        self.data[yoffset + x].$op_assign(rhs[y]);
                    }
                }
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<Vector<X, T>> for ByRowMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Vector<X, T>) {
                self.$op_assign(&rhs);
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait_assign<&Vector<X, T>> for ByRowMut<'_, X, Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Vector<X, T>) {
                for y in 0..Y {
                    let yoffset = X * y;
                    for x in 0..X {
                        self.matrix.data[yoffset + x].$op_assign(rhs[x]);
                    }
                }
            }
        }

        // ======== Vector =====================================================
        // TODO: ByRow, ByRowMut, IntoByRow for Vector, &Vector, &mut Vector

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                (&self).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, mut rhs: Matrix<X, Y, T>) -> Self::Output {
                for y in 0..Y {
                    let yoffset = y * X;
                    for x in 0..X {
                        let index = yoffset + x;
                        rhs.data[index] = self[y].$op(rhs.data[index]);
                    }
                }

                rhs
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                self.$op(rhs.clone())
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<Matrix<X, Y, T>> for &mut Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: Matrix<X, Y, T>) -> Self::Output {
                (self as &Vector<Y, T>).$op(rhs)
            }
        }

        impl<const X: usize, const Y: usize, T: Number> $trait<&Matrix<X, Y, T>> for &mut Vector<Y, T>
        where [T; X * Y]: Sized $(, $($where)*)?
        {
            type Output = Matrix<X, Y, T>;

            #[inline]
            fn $op(self, rhs: &Matrix<X, Y, T>) -> Self::Output {
                (self as &Vector<Y, T>).$op(rhs)
            }
        }
    };
}

impl_ops!(@commutative     Add AddAssign add add_assign);
impl_ops!(@non_commutative Sub SubAssign sub sub_assign);
impl_ops!(@commutative     Mul MulAssign mul mul_assign);
impl_ops!(@non_commutative Div DivAssign div div_assign);
impl_ops!(@non_commutative Pow PowAssign pow pow_assign where T: PowAssign + Pow<Output = T>);

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
