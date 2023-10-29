use std::ops::{Add, Mul, Sub, Div, AddAssign, MulAssign, SubAssign, DivAssign};
use std::fmt::{Display, Debug};

use crate::ops::{Pow, PowAssign};

pub trait Number:
    Sized + Clone + Copy +
    Add<Output = Self> +
    Sub<Output = Self> +
    Mul<Output = Self> +
    Div<Output = Self> +
    AddAssign + SubAssign + MulAssign + DivAssign +
    Display + Debug +
    PartialEq + PartialOrd +
    Default
{
    const ZERO: Self;
    const ONE: Self;
}

impl Number for i8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for i16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for i32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for i64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for i128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for isize {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for u8 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for u16 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for u32 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for u64 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for u128 {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for usize {
    const ZERO: Self = 0;
    const ONE: Self = 1;
}

impl Number for f32 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Number for f64 {
    const ZERO: Self = 0.0;
    const ONE: Self = 1.0;
}

impl Pow<u32> for i8 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<u32> for i16 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<u32> for i32 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<u32> for i64 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<u32> for i128 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<u32> for isize {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: u32) -> Self::Output {
        self.pow(rhs)
    }
}

impl Pow<f32> for f32 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: f32) -> Self::Output {
        self.powf(rhs)
    }
}

impl Pow<i32> for f32 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: i32) -> Self::Output {
        self.powi(rhs)
    }
}

impl Pow<f64> for f64 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: f64) -> Self::Output {
        self.powf(rhs)
    }
}

impl Pow<i32> for f64 {
    type Output = Self;

    #[inline]
    fn pow(self, rhs: i32) -> Self::Output {
        self.powi(rhs)
    }
}

impl PowAssign<u32> for i8 {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<u32> for i16 {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<u32> for i32 {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<u32> for i64 {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<u32> for i128 {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<u32> for isize {
    #[inline]
    fn pow_assign(&mut self, rhs: u32) {
        *self = self.pow(rhs)
    }
}

impl PowAssign<f32> for f32 {
    #[inline]
    fn pow_assign(&mut self, rhs: f32) {
        *self = self.powf(rhs)
    }
}

impl PowAssign<i32> for f32 {
    #[inline]
    fn pow_assign(&mut self, rhs: i32) {
        *self = self.powi(rhs)
    }
}

impl PowAssign<f64> for f64 {
    #[inline]
    fn pow_assign(&mut self, rhs: f64) {
        *self = self.powf(rhs)
    }
}

impl PowAssign<i32> for f64 {
    #[inline]
    fn pow_assign(&mut self, rhs: i32) {
        *self = self.powi(rhs)
    }
}
