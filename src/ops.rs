pub trait Unit {
    fn unit() -> Self;
}

pub trait Get<Idx: ?Sized> {
    type Output: ?Sized;

    #[track_caller]
    fn get(&self, index: Idx) -> Option<&Self::Output>;
}

pub trait GetMut<Idx: ?Sized>: Get<Idx> {
    #[track_caller]
    fn get_mut(&mut self, index: Idx) -> Option<&mut Self::Output>;
}

pub trait Pow<Rhs = Self> {
    type Output;

    fn pow(self, rhs: Rhs) -> Self::Output;
}

#[inline]
pub fn pow<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where Lhs: Pow<Rhs> {
    lhs.pow(rhs)
}

pub trait PowAssign<Rhs = Self> {
    fn pow_assign(&mut self, rhs: Rhs);
}

pub trait Dot<Rhs = Self> {
    type Output;

    fn dot(self, rhs: Rhs) -> Self::Output;
}

pub trait Cross<Rhs = Self> {
    type Output;

    fn cross(self, rhs: Rhs) -> Self::Output;
}

pub trait DotAssign<Rhs = Self> {
    fn dot_assign(&mut self, rhs: Rhs);
}

pub trait CrossAssign<Rhs = Self> {
    fn cross_assign(&mut self, rhs: Rhs);
}

#[inline]
pub fn dot<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where Lhs: Dot<Rhs> {
    lhs.dot(rhs)
}

#[inline]
pub fn cross<Lhs, Rhs>(lhs: Lhs, rhs: Rhs) -> Lhs::Output
where Lhs: Cross<Rhs> {
    lhs.cross(rhs)
}

pub trait Slice<Idx: ?Sized> {
    type Output;

    fn slice(&self, index: Idx) -> Self::Output;
}

pub trait Tap {
    #[inline]
    fn tap<F>(&self, f: F)
    where F: FnOnce(&Self) {
        f(self);
    }

    #[inline]
    fn tap_mut<F>(&mut self, f: F)
    where F: FnOnce(&Self) {
        f(self);
    }
}

pub trait Pipe {
    #[inline]
    fn pipe<F, T>(&self, f: F) -> T
    where F: FnOnce(&Self) -> T {
        f(self)
    }

    #[inline]
    fn pipe_mut<F, T>(&mut self, f: F) -> T
    where F: FnOnce(&Self) -> T {
        f(self)
    }

    #[inline]
    fn into_pipe<F, T>(self, f: F) -> T
    where F: FnOnce(Self) -> T, Self: Sized {
        f(self)
    }
}
