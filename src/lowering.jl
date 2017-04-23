### Lowering

export @lower, @arrayop

import Base: eltype

"""
`Indexing(A, (idx...))`

Represents indexing over an N dimensional array A, with N `idx`
Each index could be either:
- `IndexSym{:sym}()` object: denotes iteration using `sym` as the iteration index
- `IndexConst{T}(val::T)` object: denotes a constant in that dimension
                                 (e.g.this would be wrapping an Int in case of reducedim)
"""
immutable Indexing{A, I}
    array::A
    idx::I # Tuple of Union{IndexSym, IndexConst}
end

eltype{A,I}(::Type{Indexing{A,I}}) = eltype(A)
arraytype{A,I}(::Type{Indexing{A,I}}) = A

immutable IndexSym{d} end

immutable IndexConst{T}
    val::T
end

"""
`Map(f, (arrays...))`

Represents application of function `f` on corresponding elements of `arrays`.

# Example:

`A[i]*B[j]*42` would lower to:

`Map(*, Indexing(A, IndexSym{:i}()), Indexing(B, IndexSym{:j}()), ConstArg{Int}(42))`
"""
immutable Map{F, Ts<:Tuple}
    f::F
    arrays::Ts # Tuple of Union{Indexing, ConstArg}
end

@pure function eltype{F,Ts}(::Type{Map{F,Ts}})
    promote_op_t(F, eltypes(Ts))
end

@pure function arraytype{F,Ts}(::Type{Map{F, Ts}})
    #promote_arraytype(F, map(arraytype, Ts.parameters)...)
    arraytype(Ts.parameters[1])
end

# A constant argument to Map
immutable ConstArg{T}
    val::T
end

eltype{T}(::Type{ConstArg{T}}) = T
"""
`Assign(lhs, rhs)`

represents a tensor operation. `lhs` is an `Indexing` representing the LHS of the tensor expression
`rhs` isa `Union{Indexing, Map}`
"""
immutable Assign{L<:Indexing,R,F,E}
    lhs::L
    rhs::R
    reducefn::F
    empty::E
end
Assign(lhs, rhs) = Assign(lhs, rhs, +, nothing)

function lower_index(idx, only_symbols=false)
    if isa(idx, Symbol)
        IndexSym{idx}()
    else
        if only_symbols
            throw(ArgumentError("Got $idx instead of a symbol"))
        end
        :(ArrayMeta.IndexConst($idx))
    end
end

# lower Indexing and Maps
function lower_indexing_and_maps(expr)
    @match expr begin
        A_[idx__] => :(ArrayMeta.Indexing($A, ($(map(lower_index, idx)...),)))
        f_(arg_)   => :(ArrayMeta.Map($f, ($(lower_indexing_and_maps(arg)),)))
        f_(args__)  => :(ArrayMeta.Map($f, ($(reduce(vcat, [lower_indexing_and_maps(x) for x in args])...),)))
        x_ => :(ArrayMeta.ConstArg($x))
    end
end

immutable AllocVar{sym} end # the output variable name
arraytype{A<:AllocVar}(x::Type{A}) = A
function lower_alloc_indexing(expr)
    @match expr begin
        A_[idx__] => :(ArrayMeta.Indexing(ArrayMeta.AllocVar{$(Expr(:quote, A))}(), ($(map(lower_index, idx)...),)))
        x_ => :(ArrayMeta.ConstArg($x))
    end
end

function lower(expr, reducefn, default=nothing)
    lhs, rhs, alloc = @match expr begin
        (lhs_ = rhs_) => lhs, rhs, false
        (lhs_ := rhs_) => lhs, rhs, true
        _ => error("Expression is not of the form LHS = RHS")
    end
    lidxs = indicesinvolved(lhs)
    ridxs = indicesinvolved(rhs)

    rhs_lowered = lower_indexing_and_maps(rhs)

    # which indices are reduced over
    reduceddims = setdiff(flatten(last.(ridxs)), flatten(last.(lidxs)))

    if alloc
        :(ArrayMeta.Assign($(lower_alloc_indexing(lhs)), $rhs_lowered,
                            $reducefn, $default))
    else
        :(ArrayMeta.Assign($(lower_indexing_and_maps(lhs)), $rhs_lowered,
                            $reducefn, $default))
    end
end

macro lower(expr, reducefn=+, default=nothing)
    lower(expr, reducefn, default) |> esc
end

"""
`arrayop!(t::Assign)`

Perform a tensor operation
"""
macro arrayop(expr, reducefn=+, default=nothing)
    :(ArrayMeta.arrayop!($(lower(expr, reducefn, default)))) |> esc
end

@inline function arrayop!{L,R}(t::Assign{L,R})
    arrayop!(arraytype(L), t)
end

function hasreduceddims{L,R,F,E}(op::Type{Assign{L,R,F,E}})
    rspaces = index_spaces(:(rhs), R)
    lspaces = index_spaces(:(lhs), L)
    !isempty(setdiff(keys(rspaces), keys(lspaces)))
end
hasreduceddims(op::Assign) = hasreduceddims(typeof(op))

# Kind of a hack,
# this method is the allocating version of arrayop!
@inline @generated function arrayop!{var, L,R}(::Type{AllocVar{var}}, t::Assign{L,R})

    rspaces = index_spaces(:(t.rhs), R)
    lspaces = index_spaces(:(t.lhs), L)
    reduced_dims = setdiff(keys(rspaces), keys(lspaces))
    dims = Any[]
    for (i, k) in enumerate(L.parameters[2].parameters)
        if k <: IndexSym
            sym = k.parameters[1]

            if !haskey(rspaces, sym)
                error("Could not figure out output dimension for symbol $sym.")
            end

            dim = first(rspaces[sym])
            dimsz = :(indices($(dim[3]), $(dim[2])))
            push!(dims, dimsz)
        else
            # Indexed by a constant
            push!(dims, :(indices($(indexing_expr(:(t.lhs), k, i)), 1)))
        end
    end

    if !isempty(reduced_dims)
        # this means that the operation involves some kind of
        # accumulation. We need to fill the array with an appropriate initial value
        lhsarray = quote
            init = t.empty === nothing ?
                reduction_identity(t.reducefn, $(eltype(R))) : t.empty
            lhs = ArrayMeta.allocarray($(arraytype(R)), init, $(dims...))
        end
    else
        # we don't care how the array is initialized we're going to overwrite it fully anyway.
        lhsarray = quote
            lhs = ArrayMeta.allocarray($(arraytype(R)), nothing, $(dims...))
        end
    end

    quote
        $lhsarray
        arrayop!(ArrayMeta.Assign(Indexing(lhs, t.lhs.idx), t.rhs,
                                   t.reducefn, t.empty))
    end
end

function allocarray{T,N}(::Type{Array{T,N}}, default, sz...)
    x = similar(Array{T}, sz...)
    if default !== nothing
        fill!(x, default)
    else
        x
    end
end
