use std::marker::PhantomData;

pub enum Assert<const CHECK: bool> {}

pub trait IsTrue {}

impl IsTrue for Assert<true> {}

pub struct TypeEq<A, B> {
    phantom_data: PhantomData<(A, B)>
}

impl<T> IsTrue for TypeEq<T, T> {}
