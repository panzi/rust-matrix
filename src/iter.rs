use crate::{Number, Matrix, Vector};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct RowIter<'a, const X: usize, const Y: usize, T: Number>
where [T; X * Y]: Sized {
    matrix: &'a Matrix<X, Y, T>,
    index: usize,
}

impl<'a, const X: usize, const Y: usize, T: Number> RowIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a Matrix<X, Y, T>) -> Self {
        Self {
            matrix,
            index: 0,
        }
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> std::iter::Iterator for RowIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    type Item = Vector<X, T>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= Y {
            return None;
        }

        let yoffset = self.index * X;
        let mut data = Box::new([T::default(); X]);
        data.copy_from_slice(&self.matrix.data()[yoffset..yoffset + X]);

        self.index += 1;

        Some(Vector::from(data))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(Y - self.index))
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if self.index >= Y {
            return None;
        }

        Some(self.matrix.row(Y - 1))
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        Y - self.index
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let rem = Y - self.index;
        if n >= rem {
            self.index = Y;
            return None;
        }
        self.index += n;
        self.matrix.try_row(self.index)
    }

    #[inline]
    #[cfg(target_feature = "iter_advance_by")]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let rem = Y - self.index;
        if n > rem {
            self.index = Y;
            return Err(std::num::NonZeroUsize::try_from(n - rem).unwrap());
        }
        self.index += n;
        Ok(())
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> std::iter::ExactSizeIterator for RowIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    fn len(&self) -> usize {
        Y - self.index
    }

    #[inline]
    #[cfg(target_feature = "exact_size_is_empty")]
    fn is_empty(&self) -> bool {
        self.index >= Y
    }
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct ColumnIter<'a, const X: usize, const Y: usize, T: Number>
where [T; X * Y]: Sized {
    matrix: &'a Matrix<X, Y, T>,
    index: usize,
}

impl<'a, const X: usize, const Y: usize, T: Number> ColumnIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    pub fn new(matrix: &'a Matrix<X, Y, T>) -> Self {
        Self {
            matrix,
            index: 0,
        }
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> std::iter::Iterator for ColumnIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    type Item = Vector<Y, T>;

    fn next(&mut self) -> Option<Self::Item> {
        let res = self.matrix.try_column(self.index);

        if res.is_some() {
            self.index += 1;
        }

        res
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(X - self.index))
    }

    #[inline]
    fn last(self) -> Option<Self::Item>
    where
        Self: Sized,
    {
        if self.index >= X {
            return None;
        }

        self.matrix.try_column(X - 1)
    }

    #[inline]
    fn count(self) -> usize
    where
        Self: Sized,
    {
        Y - self.index
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<Self::Item> {
        let rem = X - self.index;
        if n >= rem {
            self.index = Y;
            return None;
        }
        self.index += n;
        self.matrix.try_column(self.index)
    }

    #[inline]
    #[cfg(target_feature = "iter_advance_by")]
    fn advance_by(&mut self, n: usize) -> Result<(), NonZeroUsize> {
        let rem = X - self.index;
        if n > rem {
            self.index = Y;
            return Err(std::num::NonZeroUsize::try_from(n - rem).unwrap());
        }
        self.index += n;
        Ok(())
    }
}

impl<'a, const X: usize, const Y: usize, T: Number> std::iter::ExactSizeIterator for ColumnIter<'a, X, Y, T>
where [T; X * Y]: Sized {
    #[inline]
    fn len(&self) -> usize {
        X - self.index
    }

    #[inline]
    #[cfg(target_feature = "exact_size_is_empty")]
    fn is_empty(&self) -> bool {
        self.index >= X
    }
}
