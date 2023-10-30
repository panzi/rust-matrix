use crate::Vector;

pub trait RangeIter {
    const LEN: usize;
    type Iter: std::iter::Iterator<Item = usize> + Clone;

    fn iter(&self) -> Self::Iter;
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Range<const START: usize, const END: usize> ();

impl<const START: usize, const END: usize> Range<START, END> {
    pub const START: usize = START;
    pub const END: usize = END;

    #[inline]
    pub const fn to_range(&self) -> std::ops::Range<usize> {
        START..END
    }
}

impl<const START: usize, const END: usize> From<Range<START, END>> for std::ops::Range<usize> {
    #[inline]
    fn from(value: Range<START, END>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize> From<&Range<START, END>> for std::ops::Range<usize> {
    #[inline]
    fn from(value: &Range<START, END>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize> From<&mut Range<START, END>> for std::ops::Range<usize> {
    #[inline]
    fn from(value: &mut Range<START, END>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize> RangeIter for Range<START, END> {
    const LEN: usize = END - START;
    type Iter = std::ops::Range<usize>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        START..END
    }

    #[inline]
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize> {
        let mut index = START;
        let data = Box::new([(); Self::LEN].map(|_| {
            let value = index;
            index += 1;
            value
        }));

        Vector::from(data)
    }
}

impl<const START: usize, const END: usize> std::iter::IntoIterator for Range<START, END> {
    type Item = usize;
    type IntoIter = std::ops::Range<usize>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a, const LEN: usize> RangeIter for &'a [usize; LEN] {
    const LEN: usize = LEN;
    type Iter = std::iter::Cloned<std::slice::Iter<'a, usize>>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        <[usize]>::iter(*self).cloned()
    }

    #[inline]
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize> {
        // XXX: Why is this transmute necesarry? Compiler bug? Why doesn't the compiler know that LEN == Self::LEN?
        let data: &[usize; Self::LEN] = unsafe { std::mem::transmute(*self) };
        Vector::from(data)
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

    #[inline]
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize> {
        Vector::from([*self])
    }
}
