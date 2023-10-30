Rust Matrix
===========

A tiny **untested**, unoptimized and incomplete Matrix library that I wrote for
fun and learning.

Also note: It uses unsafe to do things like transmuting between `Matrix<X, Y, T>`
and `Vector<X * Y, T>` or transmuting between `[[T; X]; Y]` and `[T; X * Y]`.
This part *really* needs testing.
