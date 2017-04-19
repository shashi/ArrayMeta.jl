# ArrayMeta

## Rationale

The abstractness (or lack thereof) of fallback implementations of array operations in Base Julia left me unsatisfied. Some of the problems we have:

1. Fallback methods of array operations assume the presence of an efficient elementwise access (`getindex`) operation (either `IndexLinear` or `IndexCartesian`). It is hard to reconcile this abstraction for distributed / blocked arrays, resulting in DArray implementations having to wrap every operation on AbstractArray. Wrapping some of these operations is trivial (e.g. `map` and `reduce`) but some are not trivial to wrap (e.g. `reducedim`, `broadcast`). Further, the wrappers need to be constantly updated to keep in sync with Base's set of features.
2. Operations often involve similar boiler plate code for dimensionality checks and reflection to find output array type and dimensions.
3. Not all operations are optimized for memory locality, those that are have different implementations - thus leading to more complex code that need not exist.

This package aims to evolve the means available to express array operations at a higher level than currently possible.

## The `@arrayop` macro

The `@arrayop` macro can express array operations by denoting how the dimensions of the input array interact to produce dimensions of the output array. By example,

```julia
using Base.Test

X = convert(Array, reshape(1:12, 4,3))
Y = ones(3,4)

# copy
@test @arrayop(_[i,j] := X[i,j]) == X

# transpose
@test @arrayop(_[i,j] := X[j,i]) == X'

# elementwise 1-arg
@test @arrayop(_[i,j] := -X[i,j]) == -X

# elementwise 2-args
@test @arrayop(_[i,j] := X[i,j] + Y[j,i]) == X + Y'

# elementwise with const
@test @arrayop(_[] := 2 * X[i,j])[] == sum(2.*X)

# reduce default (+)
@test @arrayop(_[] := X[i,j])[] == sum(X)

# reduce with function
@test @arrayop(_[] := X[i,j], [i=>*, j=>*])[] == prod(X)

# reducedim default (+)
@test @arrayop(_[1, j] := X[i,j]) == sum(X, 1)
@test @arrayop(_[i, 1] := X[i,j]) == sum(X, 2)

# reducedim with custom function
@test @arrayop(_[1, j] := X[i,j], [i=>*]) == prod(X, 1)

# broadcast
y = [1, 2, 3, 4]
@test @arrayop(_[i, j] := X[i, j] + y[i]) == X .+ y
y = [1 2 3]
@test @arrayop(_[i, j] := X[i, j] + y[1, j]) == X .+ y

# matmul
@test @arrayop(_[i, j] := X[i,k] * Y[k,j]) == X*Y
```

The same expressions currently [work on Dagger arrays](https://github.com/shashi/ArrayMeta.jl/blob/d1aced541e82de5021ed92ea72f29375b472c77c/test/runtests.jl#L165-L210).

`@arrayop Z[i, j] = X[i,k] * Y[k,j]` works in-place (if `=` is used instead of `:=`) by overwriting `Z`.

You can imagine Base defining the `reducedim` function as:

```julia
@generated function reducedim{dim}(f, X::AbstractArray, ::Val{dim})
    idx_in  = Any[Symbol("i$n") for n=1:ndims(X)]
    idx_out = copy(idx_in)
    idx_out[dim] = 1
    :(@arrayop _[$(idx_out...)] := X[$(idx_in...)])
end
```

## How it works

### Step 1: Lowering an `@arrayop` expression to an intermediate form

The `@arrayop <expr>` simply translates to `arrayop!(@lower <expr>)`. The goal of `@lower` is to lower the expression to type `ArrayOp` which contains an LHS and an RHS consisting of combination of `Indexing`, `Map` and `Reduce` types.

- `A[i, j]` becomes `Indexing(A, IndexSym{:i}(), IndexSym{:j}())`
- `A[i, 1]` becomes `Indexing(A, IndexSym{:i}(), IndexConst{Int}(1))`
- `f(A[i, j])` becomes `Map(f, <lowering of A[i,j]>)`
- `B[i] = f(A[i, j])` creates an ArrayOp with `Reduce(f, <lowering of f(A[i,j])>, reduction_identity(f, eltype(A)))` on the RHS, and `Indexing(B, IndexSym{:i}())` on LHS
- `B[i] := f(A[i, j])` creates a similar `ArrayOp` but the LHS contains `Indexing(AllocVar{:B}, IndexSym{:i}())` denoting an output array (named `B`) should be allocated and then iterated.

You can try out a few expressions with the `@lower` macro to see how they get lowered. These types for representing the lowered form of the expression are parameterized by the types of all their arguments allowing functions in the next steps to dispatch on and generate efficient code tailored to the specific expression and specific array types.

### Step 2: calling `arrayop!` - entry point to the `@generated` world

The lowered object got from the expression in Step 1 is passed to `arrayop!`. By default `arrayop!(op)` calls `arrayop!(<type of LHS array>, op)` which is a way to dispatch to different implementations based on the type of array in the LHS. This is how `@arrayop` is able to pick different implementations for normal arrays and Dagger arrays. `arrayop!(::Type{AllocVar}, op)` is special and tries to allocate the LHS [by calling `ArrayMeta.allocarray`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/lowering.jl#L202) and then calls `arrayop!(<type of LHS>, op)`.

### Step 3: generating loop expressions

The task of `arrayop!` is to act as a generated function which returns the code that can perofm a given operation. The code for doing this on AbstractArrays is at [`src/impl/abstract.jl`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/impl/abstract.jl#L124-L126). The Dagger implementation is at [`src/impl/dagger.jl`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/impl/dagger.jl#L42-L49). It was possible to acheive the Dagger implementation without generating any loop expressions, and interestingly, only by rewriting the lowered form from Step 1 to a lowered form that act on the chunks of the DArray can be handled by the AbstractArray implementation.
