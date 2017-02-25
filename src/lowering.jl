import Base: eltype

"""
`Iter(A, (idx...))`

Represents iteration over an N dimensional array A, with N `idx`
Each index could be either:
- `IterSym{:sym}()` object: denotes iteration using `sym` as the iteration index
- `IterConst{T}(val::T)` object: denotes a constant in that dimension
                                 (e.g.this would be wrapping an Int in case of reducedim)
"""
immutable Iter{A, I}
    A::A
    idx::I # Tuple of Union{IterSym, IterConst}
end

eltype{A,I}(::Type{Iter{A,I}}) = eltype(A)
arraytype{A,I}(::Type{Iter{A,I}}) = A

immutable IterSym{d} end

immutable IterConst{T}
    val::T
end


@testset "metadata about Iter" begin
    itr = Iter(rand(10,10), (IterSym{:i}(), IterSym{:j}()))
    @test eltype(typeof(itr)) == Float64
    @test arraytype(typeof(itr)) == Array{Float64,2}
end

###### Map ######

"""
`Map(f, (Xs...))`

Represents application of function `f` on `Iter` or `ConstArg` objects in `Xs`.

For example `A[i]*B[j]*42` would lower to:

`Map(*, Iter(A, IterSym{:i}()), Iter(B, IterSym{:j}()), ConstArg{Int}(42))`
"""
immutable Map{F, Ts<:Tuple}
    f::F
    Xs::Ts # Tuple of Union{Iter, ConstArg}
end

function eltype{F,Ts}(::Type{Map{F,Ts}})
    Base.@_pure_meta
    promote_op_t(F, eltypes(Ts))
end

function arraytype{F,Ts}(::Type{Map{F, Ts}})
    Base.@_pure_meta
    promote_arraytype(F, Ts)
end

# A constant argument. We keep track of the type.
immutable ConstArg{T}
    val::T
end

eltype{T}(::Type{ConstArg{T}}) = T


@testset "metadata about Map" begin
    itr1 = Iter(rand(Int,10), (IterSym{:i}(), IterSym{:j}()))
    itr2 = Iter(rand(10), (IterSym{:i}(), IterSym{:j}()))
    map1 = Map(*, (itr1, itr2))
    map2 = Map(/, (ConstArg(1), itr1))
    @test eltype(typeof(map1)) == Float64
    @test eltype(typeof(map2)) == Float64
    @test arraytype(typeof(map1)) == Array{Float64,1}
end


###### Reduce ######

"""
`Reduce(idx::IterSym, f, X, empty=default_identity)`

Represents reduction of dimension indexed by `idx` in `X` using
the function `f`, and `empty` as the identity value.

`X` isa `Union{Iter, Map, Reduce}`
"""
immutable Reduce{idx<:IterSym, F, T, E}
    f::F
    X::T
    empty::E
end

function Reduce{I<:IterSym,F,T}(dim::I, f::F, X::T, empty=reduce_identity(f, eltype(T)))
    Reduce{I,F,T, typeof(empty)}(f,X,empty)
end

eltype{idx,F,T,E}(::Type{Reduce{idx, F, T, E}}) = _promote_op_t(F, E, eltype(T))
function arraytype{idx,F,T,E}(::Type{Reduce{idx,F,T,E}})
    # this is of course not correct.
    # but arraytype must only be used on the array type rather than the
    # element type. So this is OK.
    arraytype(T)
end

@testset "Reduce meta" begin
    itr1 = Iter(rand(Int,10), (IterSym{:i}(),))
    @test eltype(Reduce(IterSym{:i}(), push!, itr1, Int[])) == Array{Int,1}
end


"""
`TensorOp(lhs, rhs)`

represents a tensor operation. `lhs` is an `Iter` representing the LHS of the tensor expression
`rhs` isa `Union{Iter, Map, Reduce}`
"""
immutable TensorOp{L<:Iter,R}
    lhs::L
    rhs::R
end


### Lowering a tensor operation expression to Iter, Map, Reduce ###

function lower_index(idx, only_symbols=false)
    if isa(idx, Symbol)
        IterSym{idx}()
    else
        if only_symbols
            throw(ArgumentError("Got $idx instead of a symbol"))
        end
        :(IterConst($idx))
    end
end

# lower Iter and Maps
function lower_iter_maps(expr)
    @match expr begin
        A_[idx__] => :(Iter($A, ($(map(lower_index, idx)...),)))
        f_(arg_)   => :(Map($f, ($(lower_iter_maps(arg)),)))
        f_(args__)  => :(Map($f, ($(reduce(vcat, [lower_iter_maps(x) for x in args])...),)))
        x_ => :(ConstArg($x))
    end
end

# Get a Dictionary of reduction functions
function reduction_functions(reductions)
    @match reductions begin
        (i_=>f_) => Dict(i => f)
        [R__] => reduce(merge, map(reduction_functions, R))
        nothing => Dict()
        _ => error("Invalid reduction spec")
    end
end

function lower(expr, reductions)
    lhs, rhs = @match expr begin
        (lhs_ = rhs_) => lhs, rhs
        _ => error("Expression is not of the form LHS = RHS")
    end
    lidxs = indicesinvolved(lhs)
    ridxs = indicesinvolved(rhs)

    # which indices iterate over which dimension of the input
    # idxdims = index_dim_map(ridxs) # TODO: use this to verify correct dimensions
    lowered_maps = lower_iter_maps(rhs)

    # which indices are reduced over
    reduceddims = setdiff(flatten(cdr.(ridxs)), flatten(cdr.(lidxs)))
    reduce_dict = reduction_functions(reductions)

    # lower reduces
    rhs_lowered = reduce(lowered_maps, reduceddims) do ex, idx
        :(Reduce($(lower_index(idx, true)), $(get(reduce_dict, idx, +)), $ex))
    end

    :(TensorOp($(lower_iter_maps(lhs)), $rhs_lowered))
end

macro lower(expr, reductions=:nothing)
    lower(expr, reductions) |> esc
end

@testset "Lower" begin
    A = rand(2,2); B = rand(2,2); C = rand(2,2);
    i, j, k = [IterSym{x}() for x in [:i,:j,:k]]
    # map
    @test @lower(A[i,j] = B[i,j]) == TensorOp(Iter(A, (i, j)), Iter(B, (i, j)))

    # transpose
    @test @lower(A[i,j] = B[j,i]) == TensorOp(Iter(A, (i, j)), Iter(B, (j, i)))

    # reduced over i:
    @test @lower(A[j] = B[j,i])   == TensorOp(Iter(A, (j,)), Reduce(i, +, Iter(B, (j, i))))

    # reduced over i, output is reducedim
    @test @lower(A[1,j] = B[i,j]) == TensorOp(Iter(A, (IterConst{Int}(1), j)), Reduce(i, +, Iter(B, (i, j))))

    # reduce both dimensions, use * to reduce i and + to reduce j
    @test @lower(A[1,1] = B[i,j], [i=>*,j=>+]) == TensorOp(Iter(A, (IterConst{Int}(1), IterConst{Int}(1))),
                                                           Reduce(j, +, Reduce(i, *, Iter(B, (i, j)))))
end
