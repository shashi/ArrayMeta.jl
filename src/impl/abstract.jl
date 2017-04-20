using TiledIteration

# constructs Dict
# Keys:  indexing symbol
# Value: a list of (array type, dimension, expr) tuples which correspond to that
#        indexing symbol
function index_spaces{X,Idx}(name, itr::Type{Indexing{X, Idx}})
    idx_spaces = Dict()

    for (dim, idx) in enumerate(Idx.parameters)
        j = indexing_expr(name, idx, dim)
        if isa(j, Symbol)
            # store mapping from index symbol to "index space" of arrays
            # being iterated over
            Base.@get! idx_spaces j []
            push!(idx_spaces[j], (X, dim, :($name.array)))
        end
    end

    idx_spaces
end

indexing_expr{x}(name, ::Type{IndexSym{x}}, i) = x
indexing_expr{C<:IndexConst}(name, ::Type{C}, i) = :($name.idx[$i].val)

function index_spaces{F, X}(name, itr::Type{Map{F, X}})
    inner = [index_spaces(:($name.arrays[$i]), idx)
             for (i, idx) in enumerate(X.parameters)]
    merge_dictofvecs(inner...)
end

function index_spaces{I,F,T,E}(name, itr::Type{Reduce{I,F,T,E}})
    index_spaces(:($name.array), T)
end

function index_spaces{L,R,F,E}(name, itr::Type{ArrayOp{L,R,F,E}})
    merge_dictofvecs(index_spaces(:($name.lhs), L), index_spaces(:($name.rhs), R))
end

function index_space_iterator{A<:AbstractArray}(T::Type{A}, dimension, name)
    :(1:size($name, $dimension))
end

### Construction of loop expressions in type domain
### This is the fallback implementation for AbstractArrays

function get_subscripts{X,Idx}(name, itr::Type{Indexing{X, Idx}})
    [indexing_expr(name, idx, i) for (i, idx) in  enumerate(Idx.parameters)]
end

# Generate the expression corresponding to a type
function kernel_expr(name, lhs, itr)
    kernel_expr(name, arraytype(itr), lhs, itr)
end

function kernel_expr{X, Idx, A<:AbstractArray}(name, ::Type{A},
                                               lhs,
                                               itr::Type{Indexing{X, Idx}})
    idx = get_subscripts(name, itr)
    :($name.array[$(idx...)])
end

function kernel_expr{A<:AbstractArray, F, Ts}(name, ::Type{A},
                                              lhs,
                                              itr::Type{Map{F, Ts}})

    inner_kernels = [kernel_expr(:($name.arrays[$i]), arraytype(T), lhs, T)
                        for (i, T) in enumerate(Ts.parameters)]

    :($name.f($(inner_kernels...)))
end

function kernel_expr{A <: AbstractArray, idx, F, T, E}(name, ::Type{A},
                                   lhs,
                                   itr::Type{Reduce{IndexSym{idx}, F, T, E}})

    inner = kernel_expr(:($name.array), arraytype(T), lhs, T)
    :($name.f($lhs, $inner))
end

allequal(x) = true
function allequal(x, xs...)
    x == xs[1] && allequal(xs...)
end

function arrayop_body{A<:AbstractArray, L,R,F,E}(name, ::Type{A}, op::Type{ArrayOp{L,R,F,E}})
    acc = kernel_expr(:($name.lhs), :(), L) # :() will be ignored
    rhs_inner = kernel_expr(:($name.rhs), acc, R)

    expr = :($acc = $rhs_inner)
    checks = :()
    input_ranges = Any[]

    simded = false

    for (sym, spaces) in index_spaces(name, op) # sort this based on # of potential cache misses
        if length(spaces) > 1
            # check dimensions for equality
            equal_dims = [:(indices($(d[3]), $(d[2]))) for d in spaces]
            checks = :($checks; @assert allequal($(equal_dims...),))
        end
        T,dim,nm = first(spaces)
        sym_range = Symbol("$(sym)_range")
        push!(input_ranges, sym_range => :(indices($nm, $dim)))
        expr = :(for $sym in $sym_range
                    $expr
                end)
        if !simded
            # innermost loop
            expr = :(@simd $expr)
            simded = true
        end
    end

    expr = quote
        ranges = ($(map(last, input_ranges)...),)
        @inbounds for tile in TileIterator(ranges, tilesize(ranges))
            ($(map(first, input_ranges)...),) = tile
            $expr
        end
    end

    :($checks; $expr; $name.lhs.array)
end

function tilesize(ranges)
    map(x->16, ranges)
end

@inline @generated function arrayop!{L,R,A<:AbstractArray}(::Type{A}, t::ArrayOp{L,R})
    arrayop_body(:t, arraytype(L), t)
end
