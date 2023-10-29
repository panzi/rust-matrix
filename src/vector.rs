use std::ops::{Add, Mul, Neg, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign, Index, IndexMut};
use std::iter::{Sum, Product, IntoIterator};
use std::fmt::{Display, Debug};

use crate::Matrix;
use crate::assert::{IsTrue, Assert};
use crate::number::Number;
use crate::ops::{Get, GetMut, Pow, PowAssign, Unit, Slice};
use crate::range::{RangeIter, Range};

#[repr(transparent)]
#[derive(Clone)]
pub struct Vector<const N: usize, T: Number=f64> {
    data: Box<[T; N]>
}

impl<const N: usize, T: Number> Vector<N, T> {
    pub const N: usize = N;
    pub const SHAPE: [usize; 1] = [N];

    #[inline]
    pub fn iter(&self) -> impl std::iter::Iterator<Item = &T> {
        self.data.iter()
    }

    #[inline]
    pub fn iter_mut(&mut self) -> impl std::iter::Iterator<Item = &mut T> {
        self.data.iter_mut()
    }

    #[inline]
    pub fn data(&self) -> &[T; N] {
        &self.data
    }

    #[inline]
    pub fn data_mut(&mut self) -> &mut [T; N] {
        &mut self.data
    }

    #[inline]
    pub fn into_data(self) -> [T; N] {
        *self.data
    }

    #[inline]
    pub fn to_matrix<const X: usize, const Y: usize>(&self) -> Matrix<X, Y, T>
    where [T; X * Y]: Sized, Assert<{ N == X * Y }>: IsTrue {
        self.clone().into_matrix()
    }

    #[inline]
    pub fn into_matrix<const X: usize, const Y: usize>(self) -> Matrix<X, Y, T>
    where [T; X * Y]: Sized, Assert<{ N == X * Y }>: IsTrue {
        // XXX: is this correct?
        unsafe { std::mem::transmute(self) }
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Vector<N, U>
    where F: FnMut(T) -> U, U: Number {
        Vector { data: Box::new(self.data.map(f)) }
    }

    #[inline]
    pub const fn range(&self) -> Range::<0, N> {
        Range::<0, N>()
    }
}

impl<const N: usize, T: Number> IntoIterator for Vector<N, T> {
    type Item = T;
    type IntoIter = std::array::IntoIter<T, N>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<const N: usize, T: Number> AsRef<Self> for Vector<N, T>
where [T; N]: Sized
{
    #[inline]
    fn as_ref(&self) -> &Self {
        self
    }
}

impl<const N: usize, T: Number> AsRef<[T; N]> for Vector<N, T>
where [T; N]: Sized
{
    #[inline]
    fn as_ref(&self) -> &[T; N] {
        &self.data
    }
}

impl<const N: usize, T: Number> Unit for Vector<N, T> {
    #[inline]
    fn unit() -> Self {
        Self { data: Box::new([T::ONE; N]) }
    }
}

impl<const N: usize, T: Number> Default for Vector<N, T> {
    #[inline]
    fn default() -> Self {
        Self { data: Box::new([T::default(); N]) }
    }
}

impl<const N: usize, T: Number> Display for Vector<N, T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt(&'[', f)?;
        let mut iter = self.data.iter();
        if let Some(first) = iter.next() {
            Debug::fmt(first, f)?;
            for item in iter {
                Display::fmt("  ", f)?;
                Debug::fmt(item, f)?;
            }
        }
        Display::fmt(&']', f)
    }
}

impl<const N: usize, T: Number> Debug for Vector<N, T> {
    #[inline]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Display::fmt("Vector::new(", f)?;
        self.data.fmt(f)?;
        Display::fmt(&')', f)
    }
}

// ======== Equality ===========================================================

impl<const N: usize, T: Number> PartialEq for Vector<N, T>
where T: PartialEq, [T; N]: Sized {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<const N: usize, T: Number> PartialEq<[T; N]> for Vector<N, T>
where T: PartialEq, [T; N]: Sized {
    #[inline]
    fn eq(&self, other: &[T; N]) -> bool {
        &*self.data == other
    }
}

impl<const N: usize, T: Number> PartialEq<&[T; N]> for Vector<N, T>
where T: PartialEq, [T; N]: Sized {
    #[inline]
    fn eq(&self, other: &&[T; N]) -> bool {
        &*self.data == *other
    }
}

impl<const N: usize, T: Number> PartialEq<Vector<N, T>> for &[T; N]
where T: PartialEq, [T; N]: Sized {
    #[inline]
    fn eq(&self, other: &Vector<N, T>) -> bool {
        *self == &*other.data
    }
}

impl<const N: usize, T: Number> PartialEq<Vector<N, T>> for [T; N]
where T: PartialEq, [T; N]: Sized {
    #[inline]
    fn eq(&self, other: &Vector<N, T>) -> bool {
        self == &*other.data
    }
}

impl<const N: usize, T: Number> Eq
for Vector<N, T>
where T: Eq {}

// ======== From ===============================================================

impl<const N: usize, T: Number> From<[T; N]> for Vector<N, T> {
    #[inline]
    fn from(value: [T; N]) -> Self {
        Self { data: Box::new(value) }
    }
}

impl<const N: usize, T: Number> From<&[T; N]> for Vector<N, T> {
    #[inline]
    fn from(value: &[T; N]) -> Self {
        Self { data: Box::new(*value) }
    }
}

impl<const N: usize, T: Number> From<Box<[T; N]>> for Vector<N, T> {
    #[inline]
    fn from(value: Box<[T; N]>) -> Self {
        Self { data: value }
    }
}

impl<const N: usize, T: Number> From<&Box<[T; N]>> for Vector<N, T> {
    #[inline]
    fn from(value: &Box<[T; N]>) -> Self {
        Self { data: value.clone() }
    }
}

impl<const N: usize, T: Number> From<Vector<N, T>> for [T; N] {
    #[inline]
    fn from(value: Vector<N, T>) -> Self {
        *value.data
    }
}

impl<const N: usize, T: Number> From<Vector<N, T>> for Box<[T; N]> {
    #[inline]
    fn from(value: Vector<N, T>) -> Self {
        value.data
    }
}

impl<const N: usize, T: Number> From<&Vector<N, T>> for [T; N] {
    #[inline]
    fn from(value: &Vector<N, T>) -> Self {
        *value.data
    }
}

impl<const N: usize, T: Number> From<&Vector<N, T>> for Box<[T; N]> {
    #[inline]
    fn from(value: &Vector<N, T>) -> Self {
        value.data.clone()
    }
}

impl<const N: usize, T: Number> From<T> for Vector<N, T> {
    #[inline]
    fn from(value: T) -> Self {
        Self { data: Box::new([value; N]) }
    }
}

impl<const N: usize, T: Number> From<&T> for Vector<N, T> {
    #[inline]
    fn from(value: &T) -> Self {
        let value = *value;
        Self { data: Box::new([value; N]) }
    }
}

impl<const N: usize, T: Number> Index<usize> for Vector<N, T> {
    type Output = T;

    #[inline]
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<const N: usize, T: Number> IndexMut<usize> for Vector<N, T> {
    #[inline]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<const N: usize, T: Number> Get<usize> for Vector<N, T> {
    type Output = T;

    #[inline]
    fn get(&self, index: usize) -> Option<&Self::Output> {
        self.data.get(index)
    }
}

impl<const N: usize, T: Number> GetMut<usize> for Vector<N, T> {
    #[inline]
    fn get_mut(&mut self, index: usize) -> Option<&mut Self::Output> {
        self.data.get_mut(index)
    }
}

// ======== Slice ==============================================================

impl<const N: usize, T: Number, Range: RangeIter> Slice<Range> for Vector<N, T>
where [T; N]: Sized, [T; Range::LEN]: Sized
{
    type Output = Vector<{ Range::LEN }, T>;

    #[inline]
    fn slice(&self, range: Range) -> Self::Output {
        let mut iter = range.iter();
        let data = Box::new([(); Range::LEN].map(|_| self.data[iter.next().unwrap()]));

        Vector { data }
    }
}

impl<const N: usize, T: Number, const M: usize> Slice<[usize; M]> for Vector<N, T>
where [T; N]: Sized, [T; M]: Sized
{
    type Output = Vector<M, T>;

    #[inline]
    fn slice(&self, range: [usize; M]) -> Self::Output {
        let data = Box::new(range.map(|index| self.data[index]));

        Vector { data }
    }
}

impl<const N: usize, T: Number, const M: usize> Slice<&mut [usize; M]> for Vector<N, T>
where [T; N]: Sized, [T; M]: Sized
{
    type Output = Vector<M, T>;

    #[inline]
    fn slice(&self, range: &mut [usize; M]) -> Self::Output {
        let data = Box::new(range.map(|index| self.data[index]));

        Vector { data }
    }
}

// ======== Arithmetic Operations ==============================================

macro_rules! impl_ops {
    (@move_rhs @commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const N: usize, T: Number> $trait<Vector<N, T>> for &Vector<N, T>
        $(where $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, mut rhs: Vector<N, T>) -> Self::Output {
                rhs.$op_assign(self);

                rhs
            }
        }
    };

    (@move_rhs @non_commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const N: usize, T: Number> $trait<Vector<N, T>> for &Vector<N, T>
        $(where $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: Vector<N, T>) -> Self::Output {
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
        impl<const N: usize> $trait<Vector<N, $type>> for $type
        $(where $($where)*)?
        {
            type Output = Vector<N, $type>;

            #[inline]
            fn $op(self, mut rhs: Vector<N, $type>) -> Self::Output {
                rhs.$op_assign(self);

                rhs
            }
        }
    };

    (@primitive $type:ident @non_commutative $trait:ident $trait_assign:ident $op:ident $op_assign:ident $(where $($where:tt)*)?) => {
        impl<const N: usize> $trait<Vector<N, $type>> for $type
        $(where $($where)*)?
        {
            type Output = Vector<N, $type>;

            #[inline]
            fn $op(self, rhs: Vector<N, $type>) -> Self::Output {
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

        impl<const N: usize, T: Number> $trait for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const N: usize, T: Number> $trait<&Vector<N, T>> for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: &Self) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const N: usize, T: Number> $trait<T> for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Self;

            #[inline]
            fn $op(mut self, rhs: T) -> Self::Output {
                self.$op_assign(rhs);

                self
            }
        }

        impl<const N: usize, T: Number> $trait_assign for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: Self) {
                self.$op_assign(&rhs);
            }
        }

        impl<const N: usize, T: Number> $trait_assign<&Vector<N, T>> for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &Self) {
                for index in 0..N {
                    self.data[index].$op_assign(rhs.data[index]);
                }
            }
        }

        impl<const N: usize, T: Number> $trait_assign<T> for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: T) {
                for index in 0..N {
                    self.data[index].$op_assign(rhs);
                }
            }
        }

        impl<const N: usize, T: Number> $trait_assign<&T> for Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            #[inline]
            fn $op_assign(&mut self, rhs: &T) {
                let rhs = *rhs;
                for index in 0..N {
                    self.data[index].$op_assign(rhs);
                }
            }
        }

        // ======== ref ========================================================

        impl<const N: usize, T: Number> $trait<&Vector<N, T>> for &Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: &Vector<N, T>) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);

                res
            }
        }

        impl_ops!(@move_rhs @$commutative $trait $trait_assign $op $op_assign $(where $($where)*)?);

        impl<const N: usize, T: Number> $trait<T> for &Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                let mut res = self.clone();
                res.$op_assign(rhs);

                res
            }
        }

        // ======== mut ref ====================================================

        impl<const N: usize, T: Number> $trait<&Vector<N, T>> for &mut Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: &Vector<N, T>) -> Self::Output {
                (self as &Vector<N, T>).$op(rhs)
            }
        }

        impl<const N: usize, T: Number> $trait<Vector<N, T>> for &mut Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: Vector<N, T>) -> Self::Output {
                (self as &Vector<N, T>).$op(rhs)
            }
        }

        impl<const N: usize, T: Number> $trait<T> for &mut Vector<N, T>
        where [T; N]: Sized $(, $($where)*)?
        {
            type Output = Vector<N, T>;

            #[inline]
            fn $op(self, rhs: T) -> Self::Output {
                (self as &Vector<N, T>).$op(rhs)
            }
        }
    };
}

impl_ops!(@commutative     Add AddAssign add add_assign);
impl_ops!(@non_commutative Sub SubAssign sub sub_assign);
impl_ops!(@commutative     Mul MulAssign mul mul_assign);
impl_ops!(@non_commutative Div DivAssign div div_assign);
impl_ops!(@non_commutative Pow PowAssign pow pow_assign where T: PowAssign);

// ======== Pow ================================================================

macro_rules! impl_pow {
    ($type:ident $($cast:tt)*) => {
        impl<const N: usize> Pow<Vector<N, Self>> for $type
        where [Self; N]: Sized
        {
            type Output = Vector<N, Self>;

            #[inline]
            fn pow(self, mut rhs: Vector<N, Self>) -> Self::Output {
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

impl<const N: usize> Pow<&Vector<N, Self>> for f32
where [Self; N]: Sized
{
    type Output = Vector<N, Self>;

    #[inline]
    fn pow(self, rhs: &Vector<N, Self>) -> Self::Output {
        let mut res: Vector<N, Self> = self.into();
        res.pow_assign(rhs);

        res
    }
}

impl<const N: usize> Pow<&Vector<N, Self>> for f64
where [Self; N]: Sized
{
    type Output = Vector<N, Self>;

    #[inline]
    fn pow(self, rhs: &Vector<N, Self>) -> Self::Output {
        let mut res: Vector<N, Self> = self.into();
        res.pow_assign(rhs);

        res
    }
}

// ======== Neg ================================================================

impl<const N: usize, T: Number> Neg for Vector<N, T>
where [T; N]: Sized
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

impl<const N: usize, T: Number> Neg for &Vector<N, T>
where [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector {
            data: Box::new(self.data.map(|value| -value))
        }
    }
}

impl<const N: usize, T: Number> Neg for &mut Vector<N, T>
where [T; N]: Sized
{
    type Output = Vector<N, T>;

    #[inline]
    fn neg(self) -> Self::Output {
        Vector {
            data: Box::new(self.data.map(|value| -value))
        }
    }
}

// ======== Sum ================================================================

impl<const N: usize, T: Number> Sum for Vector<N, T>
where [T; N]: Sized
{
    fn sum<I: Iterator<Item = Self>>(mut iter: I) -> Self {
        if let Some(mut res) = iter.next() {
            for item in iter {
                res += item;
            }
            return res;
        }
        Vector::default()
    }
}

impl<'a, const N: usize, T: Number> Sum<&'a Vector<N, T>> for Vector<N, T>
where [T; N]: Sized
{
    fn sum<I: Iterator<Item = &'a Self>>(mut iter: I) -> Self {
        if let Some(first) = iter.next() {
            let mut res = first.clone();
            for item in iter {
                res += item;
            }
            return res;
        }
        Vector::default()
    }
}

// ======== Product ============================================================

impl<const N: usize, T: Number> Product for Vector<N, T>
where [T; N]: Sized
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

impl<'a, const N: usize, T: Number> Product<&'a Vector<N, T>> for Vector<N, T>
where [T; N]: Sized
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
