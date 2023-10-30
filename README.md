Rust Matrix
===========

A tiny **untested**, unoptimized and incomplete Matrix library that I wrote for
fun and learning.

Also note: It uses unsafe to do things like transmuting `Matrix<X, Y, T>` to
`Vector<X * Y, T>` or transmuting `[[T; X]; Y]` to `[T; X * Y]`. This part
*really* needs testing.
