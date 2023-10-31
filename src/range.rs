use crate::Vector;

// ======== RangeIter ==========================================================

pub trait RangeIter {
    const LEN: usize;
    type Iter: std::iter::Iterator<Item = usize> + Clone;

    fn iter(&self) -> Self::Iter;
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize>;
}

// ======== Range ==============================================================

#[derive(Debug, Clone, Copy)]
pub struct Range<const START: usize, const END: usize> ();

impl<const START: usize, const END: usize> Range<START, END> {
    pub const START: usize = START;
    pub const END: usize = END;

    #[inline]
    pub const fn to_range(&self) -> std::ops::Range<usize> {
        START..END
    }

    #[inline]
    pub const fn step_by<const STEP: usize>(&self) -> RangeWithStep<START, END, STEP> {
        RangeWithStep {}
    }
}

impl<const START1: usize, const END1: usize, const START2: usize, const END2: usize> PartialEq<Range<START2, END2>>
for Range<START1, END1> {
    #[inline]
    fn eq(&self, _: &Range<START2, END2>) -> bool {
        START1 == START2 && END1 == END2
    }
}

impl<const START: usize, const END: usize> Eq for Range<START, END> {}

impl<const START1: usize, const END1: usize, const START2: usize, const END2: usize> PartialOrd<Range<START2, END2>>
for Range<START1, END1> {
    #[inline]
    fn partial_cmp(&self, _: &Range<START2, END2>) -> Option<std::cmp::Ordering> {
        (START1, END1).partial_cmp(&(START2, END2))
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
        self.to_range()
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

// ======== RangeWithStep ======================================================

#[derive(Debug, Clone, Copy)]
pub struct RangeWithStep<const START: usize, const END: usize, const STEP: usize> ();

impl<const START: usize, const END: usize, const STEP: usize> RangeWithStep<START, END, STEP> {
    pub const START: usize = START;
    pub const END: usize = END;
    pub const STEP: usize = STEP;

    #[inline]
    pub fn to_range(&self) -> std::iter::StepBy<std::ops::Range<usize>> {
        (START..END).step_by(STEP)
    }
}

impl<
    const START1: usize, const END1: usize, const STEP1: usize,
    const START2: usize, const END2: usize, const STEP2: usize> PartialEq<RangeWithStep<START2, END2, STEP2>>
for RangeWithStep<START1, END1, STEP1> {
    #[inline]
    fn eq(&self, _: &RangeWithStep<START2, END2, STEP2>) -> bool {
        START1 == START2 && END1 == END2 && STEP1 == STEP2
    }
}

impl<const START: usize, const END: usize, const STEP: usize> Eq for RangeWithStep<START, END, STEP> {}

impl<
    const START1: usize, const END1: usize, const STEP1: usize,
    const START2: usize, const END2: usize, const STEP2: usize> PartialOrd<RangeWithStep<START2, END2, STEP2>>
for RangeWithStep<START1, END1, STEP1> {
    #[inline]
    fn partial_cmp(&self, _: &RangeWithStep<START2, END2, STEP2>) -> Option<std::cmp::Ordering> {
        (START1, END1, STEP1).partial_cmp(&(START2, END2, STEP2))
    }
}

impl<const START: usize, const END: usize, const STEP: usize> From<RangeWithStep<START, END, STEP>>
for std::iter::StepBy<std::ops::Range<usize>> {
    #[inline]
    fn from(value: RangeWithStep<START, END, STEP>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize, const STEP: usize> From<&RangeWithStep<START, END, STEP>>
for std::iter::StepBy<std::ops::Range<usize>> {
    #[inline]
    fn from(value: &RangeWithStep<START, END, STEP>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize, const STEP: usize> From<&mut RangeWithStep<START, END, STEP>>
for std::iter::StepBy<std::ops::Range<usize>> {
    #[inline]
    fn from(value: &mut RangeWithStep<START, END, STEP>) -> Self {
        value.to_range()
    }
}

impl<const START: usize, const END: usize, const STEP: usize> RangeIter for RangeWithStep<START, END, STEP> {
    // Overflow will give a compile error.
    const LEN: usize = (END - START + STEP - 1) / STEP; // ceiling integer division
    type Iter = std::iter::StepBy<std::ops::Range<usize>>;

    #[inline]
    fn iter(&self) -> Self::Iter {
        self.to_range()
    }

    #[inline]
    fn to_vector(&self) -> Vector<{ Self::LEN }, usize> {
        let mut index = START;
        let data = Box::new([(); Self::LEN].map(|_| {
            let value = index;
            index += STEP;
            value
        }));

        Vector::from(data)
    }
}

impl<const START: usize, const END: usize, const STEP: usize> std::iter::IntoIterator for RangeWithStep<START, END, STEP> {
    type Item = usize;
    type IntoIter = std::iter::StepBy<std::ops::Range<usize>>;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

// ======== &'a [usize; LEN] ===================================================

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

// ======== usize ==============================================================

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
