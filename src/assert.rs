pub enum Assert<const CHECK: bool> {}

pub trait IsTrue {}

impl IsTrue for Assert<true> {}

pub enum TypeEq<A, B> {}

impl<T> IsTrue for TypeEq<T, T> {}
