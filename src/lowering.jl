### Lowering

import Base: eltype

"""
`Indexing(A, (idx...))`

Represents indexing over an N dimensional array A, with N `idx`
Each index could be either:
- `IterSym{:sym}()` object: denotes iteration using `sym` as the iteration index
- `IterConst{T}(val::T)` object: denotes a constant in that dimension
                                 (e.g.this would be wrapping an Int in case of reducedim)
"""
immutable Indexing{A, I}
    array::A
    idx::I # Tuple of Union{IterSym, IterConst}
end

eltype{A,I}(::Type{Indexing{A,I}}) = eltype(A)
arraytype{A,I}(::Type{Indexing{A,I}}) = A

immutable IterSym{d} end

immutable IterConst{T}
    val::T
end

"""
`Map(f, (arrays...))`

Represents application of function `f` on corresponding elements of `arrays`.

# Example:

`A[i]*B[j]*42` would lower to:

`Map(*, Indexing(A, IterSym{:i}()), Indexing(B, IterSym{:j}()), ConstArg{Int}(42))`
"""
immutable Map{F, Ts<:Tuple}
    f::F
    arrays::Ts # Tuple of Union{Indexing, ConstArg}
end

@pure function eltype{F,Ts}(::Type{Map{F,Ts}})
    promote_op_t(F, eltypes(Ts))
end

@pure function arraytype{F,Ts}(::Type{Map{F, Ts}})
    promote_arraytype(F, map(arraytype, Ts.parameters)...)
end

# A constant argument to Map
immutable ConstArg{T}
    val::T
end

eltype{T}(::Type{ConstArg{T}}) = T

"""
`Reduce(idx::IterSym, f, array, empty=default_identity)`

Represents reduction of dimension indexed by `idx` in `array` using
the function `f`, and `empty` as the identity value.

`array` isa `Union{Indexing, Map, Reduce}`
"""
immutable Reduce{idx<:IterSym, F, T, E}
    f::F
    array::T
    empty::E
end

function Reduce{I<:IterSym,F,T}(dim::I, f::F, array::T, empty=reduction_identity(f, eltype(T)))
    Reduce{I,F,T, typeof(empty)}(f,array,empty)
end

@pure eltype{idx,F,T,E}(::Type{Reduce{idx, F, T, E}}) = _promote_op_t(F, E, eltype(T))
@pure arraytype{idx,F,T,E}(::Type{Reduce{idx, F, T, E}}) = arraytype(T)

"""
`ArrayOp(lhs, rhs)`

represents a tensor operation. `lhs` is an `Indexing` representing the LHS of the tensor expression
`rhs` isa `Union{Indexing, Map, Reduce}`
"""
immutable ArrayOp{L<:Indexing,R}
    lhs::L
    rhs::R
end

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

# lower Indexing and Maps
function lower_indexing_and_maps(expr)
    @match expr begin
        A_[idx__] => :(Indexing($A, ($(map(lower_index, idx)...),)))
        f_(arg_)   => :(Map($f, ($(lower_indexing_and_maps(arg)),)))
        f_(args__)  => :(Map($f, ($(reduce(vcat, [lower_indexing_and_maps(x) for x in args])...),)))
        x_ => :(ConstArg($x))
    end
end

# Get a Dictionary of reduction functions
function reduction_functions(reductions)
    @match reductions begin
        (i_=>f_) => Dict(i => f)
        [R__] => reduce(merge, map(reduction_functions, R))
        0 => Dict()
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
    lowered_maps = lower_indexing_and_maps(rhs)

    # which indices are reduced over
    reduceddims = setdiff(flatten(last.(ridxs)), flatten(last.(lidxs)))
    reduce_dict = reduction_functions(reductions)

    # lower reduces
    rhs_lowered = reduce(lowered_maps, reduceddims) do ex, idx
        :(Reduce($(lower_index(idx, true)), $(get(reduce_dict, idx, +)), $ex))
    end

    :(ArrayOp($(lower_indexing_and_maps(lhs)), $rhs_lowered))
end

macro lower(expr, reductions=0)
    lower(expr, reductions) |> esc
end

"""
`arrayop!(t::ArrayOp)`

Perform a tensor operation
"""
macro arrayop(expr, reductions=0)
    :(ArrayMeta.arrayop!(@lower $expr $reductions)) |> esc
end

@inline function arrayop!{L,R}(t::ArrayOp{L,R})
    arrayop!(arraytype(L), t)
end
