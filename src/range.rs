pub trait RangeIter {
    const LEN: usize;
    type Iter: std::iter::Iterator<Item = usize> + Clone;

    fn iter(&self) -> Self::Iter;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Range<const START: usize, const END: usize> ();

impl<const START: usize, const END: usize> Range<START, END> {
    pub const START: usize = START;
    pub const END: usize = END;
}

impl<const START: usize, const END: usize> RangeIter for Range<START, END> {
    const LEN: usize = END - START;
    type Iter = std::ops::Range<usize>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        START..END
    }
}

impl<'a, const LEN: usize> RangeIter for &'a [usize; LEN] {
    const LEN: usize = LEN;
    type Iter = std::iter::Cloned<std::slice::Iter<'a, usize>>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        <[usize]>::iter(*self).cloned()
    }
}

// not this because it makes an unexpected copy of the array
// impl<const LEN: usize> RangeIter for [usize; LEN] where Self: Copy {
//     const LEN: usize = LEN;
//     type Iter = std::array::IntoIter<usize, LEN>;
// 
//     #[inline]
//     fn iter(&self) -> Self::Iter {
//         self.clone().into_iter()
//     }
// }

impl RangeIter for usize {
    const LEN: usize = 1;
    type Iter = std::array::IntoIter<usize, 1>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        [*self].into_iter()
    }
}
