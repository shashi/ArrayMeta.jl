# constructs Dict
# Keys:  indexing symbol
# Value: a list of (array type, dimension, expr) tuples which correspond to that
#        indexing symbol
function index_spaces{X<:AbstractArray,Idx}(name, itr::Type{Indexing{X, Idx}})
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

function index_spaces{L,R}(name, itr::Type{ArrayOp{L,R}})
    merge_dictofvecs(index_spaces(:($name.lhs), L), index_spaces(:($name.rhs), L))
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
function kernel_expr(name, itr)
    kernel_expr(name, arraytype(itr), itr)
end

function kernel_expr{X, Idx, A<:AbstractArray}(name, ::Type{A},
                                               itr::Type{Indexing{X, Idx}},
                                               spaces=index_spaces(name, itr))
    idx = get_subscripts(name, itr)
    :($name.array[$(idx...)])
end

function kernel_expr{A<:AbstractArray, F, Ts}(name, ::Type{A},
                                              itr::Type{Map{F, Ts}},
                                              spaces=index_spaces(name, itr))

    inner_kernels = [kernel_expr(:($name.arrays[$i]), arraytype(T), T)
                        for (i, T) in enumerate(Ts.parameters)]

    :($name.f($(inner_kernels...)))
end

function kernel_expr{A <: Array, idx, F, T, E}(name, ::Type{A},
                                   itr::Type{Reduce{IndexSym{idx}, F, T, E}},
                                   spaces=index_spaces(name, itr))

    inner = kernel_expr(:($name.array), arraytype(T), T, spaces)
    !haskey(spaces, idx) && throw(ArgumentError("Reduced dimension $idx unknown"))
    iter = index_space_iterator(first(spaces[idx])...)
    quote
        let tmp = start($iter)
            if done($iter, tmp)  # 0 elements in this dimension
                acc = $name.empty # use default value
            else
                $idx, tmp = next($iter, tmp) # skip first
                acc = $inner
            end
            while !done($iter, tmp)
                $idx, tmp = next($iter, tmp)
                acc = $name.f(acc, $inner) # accumulate with f
            end
            acc # return accumulated value
        end
    end
end

allequal(x) = true
function allequal(x, xs...)
    x == xs[1] && allequal(xs...)
end

function arrayop_body{A<:AbstractArray, L,R}(name, ::Type{A}, op::Type{ArrayOp{L,R}})
    lhs_inner = kernel_expr(:($name.lhs), L)
    rhs_inner = kernel_expr(:($name.rhs), R)

    lspaces = index_spaces(:($name.lhs), L)

    expr = :($lhs_inner = $rhs_inner)
    checks = :()
    for (sym, spaces) in lspaces
        if length(spaces) > 1
            # check dimensions for equality
            equal_dims = [:(size($(d[3]), $(d[2]))) for d in spaces]
            checks = :($checks; @assert allequal($(equal_dims...),))
        end
        T,dim,nm = first(spaces)
        expr = quote
            for $sym = 1:size($nm, $dim)
                $expr
            end
        end
    end
    :($checks; $expr; $name.lhs.array)
end

@inline @generated function arrayop!{L,R,A<:AbstractArray}(::Type{A}, t::ArrayOp{L,R})
    arrayop_body(:t, arraytype(L), t)
end
