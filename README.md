# ArrayMeta

## Rationale

The abstractness (or lack thereof) of fallback implementations of array operations in Base Julia left me unsatisfied. Some of the problems we have:

1. Fallback methods of array operations assume the presence of an efficient elementwise access (`getindex`) operation (either `IndexLinear` or `IndexCartesian`). It is hard to reconcile this abstraction for distributed / blocked arrays, resulting in DArray implementations having to wrap every operation on AbstractArray. Wrapping some of these operations is trivial (e.g. `map` and `reduce`) but some are not trivial to wrap (e.g. `reducedim`, `broadcast`). Further, the wrappers need to be constantly updated to keep in sync with Base's set of features, across Julia versions.
2. Operations often involve similar boiler-plate code for dimensionality checks and reflection to find output array type and dimensions.
3. Not all operations are optimized for memory locality, those that are have different implementations - thus leading to more complex code that strictly need not exist.

This package aims to evolve the means available to express array operations at a higher level than currently possible. The basic idea is if you get the general `@arrayop` case working for a new array type then the implementations of many array operations would fall out of it (see [next section](#the-arrayop-macro) to learn about the `@arrayop` macro). `@arrayop` is also higher level than elementwise access, so distributed arrrays can implement it efficiently.

Hypothetically, if the `@arrayop` macro was moved to Base and array operations in Base like `broadcast` and `reducedim` were implemented in Base using it, then

1. We can delete a lot of array code from `Base` and replace them with much simpler `@arrayop` expressions
2. Complex array types like `DArray`, for example, wouldn't have to wrap each operation, they just need to get `@arrayop` working once, and operations defined in Base will work for that array type. (This package has a Dagger.jl array implementation as an example of this.)
3. Make optimizations (like multi-threading, memory-locality) that may speed up many operations at once across the whole array ecosystem.

## The `@arrayop` macro

The `@arrayop` macro can express array operations by denoting how the dimensions of the input arrays interact to produce dimensions of the output array. The notation is similar to [Einstein notation](https://en.wikipedia.org/wiki/Einstein_notation) (or its equivalent in [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl) see [below](#differences-with-tensoroperationsjl) for a comparison) with some added features to support more operations. The notation is best described by some examples:

```julia
X = collect(reshape(1:12, 4,3))
Y = ones(3,4)

# transpose (equivalent to X')
@arrayop Z[i,j] := X[j,i]

# elementwise 1-arg (equivalent to sin.(X))
@arrayop Z[i,j] := sin(X[i,j])

# elementwise 2-args (equivalent to X .+ Y')
@arrayop Z[i,j] := X[i,j] + Y[j,i]

# elementwise with a constant (equivalent to X .+ (im .* Y'))
@arrayop Z[i,j] := X[i,j] + im * Y[j,i]

# reduce (default +-reduce). Note: returns a 0-dimensional array
# NOTE: any dimension that is left out in the LHS of the expression
# is reduced as in Einsten notation. By default + is used as the reducer.
@arrayop Z[] := X[i,j]

# reduce with a user-specified function, * in this case.
@arrayop Z[] := X[i,j] (*)

# reducedim (defaults reducer is +)
@arrayop Z[1, j] := X[i,j] # equivalent to reducedim(+, X, 1)
@arrayop Z[i, 1] := X[i,j] # equivalent to reducedim(+, X, 2)
@arrayop Z[i] := X[i,j]    # equivalent to squeeze(reducedim(+, X, 2)) / APL-style reducedim

# reducedim with a user-specified function
@arrayop Z[1, j] := X[i,j] (*)  # equivalent to prod(X, 1)

# broadcast
y = [1, 2, 3, 4]
@arrayop Z[i, j] := X[i, j] + y[i]

y = [1 2 3]
@arrayop Z[i, j] := X[i, j] + y[1, j]

# matmul
@arrayop Z[i, j] := X[i, k] * Y[k, j]
```

Note that these expressions generate the `for`-loops to perform the required operation. They can hence be used to implement the array operations noted in the comments.

An expression like `@arrayop Z[i, j] = X[i,k] * Y[k,j]` works in-place by overwriting `Z` (notice that `=` is used instead of `:=`).

The same expressions currently work on both AbstractArrays and are specialized for efficiency to [work on Dagger arrays](https://github.com/shashi/ArrayMeta.jl/blob/d1aced541e82de5021ed92ea72f29375b472c77c/test/runtests.jl#L165-L210).

The examples here are on 1 and 2 dimensional arrays but the notation trivially generalizes to N dimensions.

As an example of how this aids genericness, potentially, Base can define the `reducedim` (for example) function on an n-dimensional array as:

```julia
@generated function reducedim{dim}(f, X::AbstractArray, ::Val{dim})
    idx_in  = Any[Symbol("i$n") for n=1:ndims(X)]
    idx_out = copy(idx_in)
    idx_out[dim] = 1
    :(@arrayop Y[$(idx_out...)] := X[$(idx_in...)], $f)
end
```

Allowing it to work efficiently both on AbstractArrays and in a specialized way on Dagger's arrays (or another array which has a specialized implementation for `@arrayop`).

## Blocked iteration

`@arrayop` uses [TiledIteration.jl](https://github.com/JuliaArrays/TiledIteration.jl) to perform operations in cache-efficient way. As a demo of this, consider a 3-dimensional `permutedims`:

```julia
julia> x = rand(128,128,128);

julia> @btime @arrayop y[i,j,k] := x[k,j,i];
  4.871 ms (16 allocations: 16.00 MiB)

julia> @btime permutedims(x, (3,2,1));
  23.611 ms (10 allocations: 16.00 MiB)
```

Presumably the Base `permutedims` doesn't make efforts to block the inputs, leading to many more cache misses than the `@arrayop` version.

```julia
julia> @btime A+A';
  3.752 ms (6 allocations: 15.26 MiB)
julia> @btime @arrayop _[i, j] := A[i,j] + A[j,i];
  2.607 ms (22 allocations: 7.63 MiB)
```

This might open up opportunities to syntactically rewrite things like `A+A'` to `@arrayop _[i,j] = A[i,j] + A[j,i]` which is faster and allocates no temporaries. This also should speed up operations on `PermuteDimsArray`.

## How it works

### Step 1: Lowering an `@arrayop` expression to an intermediate form

The `@arrayop <expr>` simply translates to `arrayop!(@lower <expr>)`. The goal of `@lower` is to lower the expression to type `Assign` which contains an LHS and an RHS consisting of combination of `Indexing` and `Map` types.

- `A[i, j]` becomes `Indexing(A, IndexSym{:i}(), IndexSym{:j}())`
- `A[i, 1]` becomes `Indexing(A, IndexSym{:i}(), IndexConst{Int}(1))`
- `f(A[i, j])` becomes `Map(f, <lowering of A[i,j]>)`
- `B[i] = f(A[i, j])` becomes `Assign(<lowering of B[i]>, <lowering of f(A[i,j])>, +, reduction_identity(+, eltype(A)))` on the RHS, and `Indexing(B, IndexSym{:i}())` on LHS
- `B[i] := f(A[i, j])` creates a similar `Assign` but the LHS contains `Indexing(AllocVar{:B}, IndexSym{:i}())` denoting an output array (named `B`) should be allocated and then iterated.

You can try out a few expressions with the `@lower` macro to see how they get lowered. These types for representing the lowered form of the expression are parameterized by the types of all their arguments allowing functions in the next steps to dispatch on and generate efficient code tailored to the specific expression and specific array types.

### Step 2: calling `arrayop!` - entry point to the `@generated` world

The lowered object got from the expression in Step 1 is passed to `arrayop!`. By default `arrayop!(op)` calls `arrayop!(<type of LHS array>, op)` which is a way to dispatch to different implementations based on the type of array in the LHS. This is how `@arrayop` is able to pick different implementations for normal arrays and Dagger arrays. `arrayop!(::Type{AllocVar}, op)` is special and tries to allocate the LHS [by calling `ArrayMeta.allocarray`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/lowering.jl#L202) and then calls `arrayop!(<type of LHS>, op)`.

### Step 3: generating loop expressions

The task of `arrayop!` is to act as a generated function which returns the code that can perofm a given operation. The code for doing this on AbstractArrays is at [`src/impl/abstract.jl`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/impl/abstract.jl#L124-L126). The Dagger implementation is at [`src/impl/dagger.jl`](https://github.com/shashi/ArrayMeta.jl/blob/7df2c0a08e3dcd6d05f1aab1dc229c925f174790/src/impl/dagger.jl#L42-L49). It was possible to acheive the Dagger implementation without generating any loop expressions, and interestingly, only by rewriting the lowered form from Step 1 to a lowered form that act on the chunks of the DArray can be handled by the AbstractArray implementation.

## Things to do

### Already practical things

- Dispatch to `BLAS.gemm!` where possible.
- Loop reordering (it does give some improvements although we do blocked iteration)
- Optimizations for PermuteDimsArray
- Communication / computation time optimizations in Dagger a la [Tensor Contraction Engine](http://www.csc.lsu.edu/~gb/TCE/)

### Researchy things

- formalize interface for new array types to implement
- implementation for sparse matrices
- implementation for IndexedTable & AxisArrays
- Autodifferentiation
- Explore more operations to express in `@arrayop`
  - mapslices
  - getindex
  - stencils

## Differences with [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)

Jutho's TensorOperations.jl has inspired this package a whole lot. However, its codebase is tailored to work specifically on tensors of real and complex numbers, their contraction, transposition, conjugation and multiplication with scalars and it does that very well. This package aims to cover all of those features in a more general framework. Notable additions:

- Works on arrays of any type
- You can use any Julia function for combining arrays and reducing dimensions, and any constants as arguments (as opposed to allowing only scalar multiplication or offsets)
- Supports indexing where some dimensions can be constants, as in:
```julia
@arrayop y[1, j] := x[i, j]
```
to support operations like `reducedim`.
- Finally, it has a dispatch system to pick different implementations for different array types. `Dagger` array operations have been implemented as an example.
