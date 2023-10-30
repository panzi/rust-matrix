use crate::{Matrix, Number, Vector, FromUSize};

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

    pub fn iter_vectors(self) -> impl std::iter::Iterator<Item = Vector<Y, T>> {
        struct Iter<const X: usize, const Y: usize, T: Number>
        where [T; X * Y]: Sized {
            data: [T; X * Y],
            x: usize
        }

        impl<const X: usize, const Y: usize, T: Number> std::iter::Iterator for Iter<X, Y, T>
        where [T; X * Y]: Sized {
            type Item = Vector<Y, T>;

            #[inline]
            fn next(&mut self) -> Option<Self::Item> {
                let x = self.x;
                if self.x >= X {
                    return None;
                }

                let mut data = Box::new([T::default(); Y]);

                for y in 0..Y {
                    data[y] = self.data[y * X + x];
                }

                self.x += 1;

                Some(Vector::from(data))
            }

            #[inline]
            fn size_hint(&self) -> (usize, Option<usize>) {
                (0, Some(X))
            }
        }

        Iter { data: self.matrix.into_data(), x: 0 }
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

    pub fn iter_vectors(&self) -> impl std::iter::Iterator<Item = Vector<Y, T>> + 'a {
        let mtx = self.matrix.data();

        (0..X).map(|x| {
            // TODO: MaybeUninit?
            let mut data = Box::new([T::default(); Y]);

            for y in 0..Y {
                data[y] = mtx[y * X + x];
            }

            Vector::from(data)
        })
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

    pub fn iter_vectors(&'a self) -> impl std::iter::Iterator<Item = Vector<Y, T>> + 'a {
        let mtx = self.matrix.data();

        (0..X).map(|x| {
            // TODO: MaybeUninit?
            let mut data = Box::new([T::default(); Y]);

            for y in 0..Y {
                data[y] = mtx[y * X + x];
            }

            Vector::from(data)
        })
    }

    #[inline]
    pub fn map<F, U>(&self, f: F) -> Matrix<Y, X, U>
    where F: FnMut(T) -> U, U: Number, [T; Y * X]: Sized {
        self.matrix.transpose_map(f)
    }
}
