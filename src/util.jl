import Base: @pure

car(x::Pair) = x.first
cdr(x::Pair) = x.second
flatten(xs) = reduce(vcat, vec.(xs))
function indicesinvolved(expr)
    @match expr begin
        A_[idx__] => [(A => idx)]
        f_(ex__)  => reduce(vcat, [indicesinvolved(x) for x in ex])
        _ => []
    end
end

let
    @test indicesinvolved(:(A[i,j,k])) == [:A=>Any[:i,:j,:k]]
    @test indicesinvolved(:(A[i,j,k]+B[x,y,z])) == [:A=>Any[:i,:j,:k], :B=>[:x,:y,:z]]
    @test indicesinvolved(:(A[i,j,k] |> f)) == [:A=>Any[:i,:j,:k]]
end

function index_dim_map(xs)
    iters = Dict()
    for (tensor, idxs) in xs
        for (i, idx) in enumerate(idxs)
            dims = Base.@get! iters idx []
            push!(dims, (tensor, i))
        end
    end
    iters
end

"""
Type Domain `promote_op`

The first argument is the type of the function
"""
@generated function promote_op_t{F,T}(::Type{F}, ::Type{T})
    :(_promote_op_t(F, $(T.parameters...)))
end

# Dummy type to imitate a splatted function application
# This is passed into Inference.return_type
immutable _F{G}
    g::G
end
(f::_F{G}){G}(x) = f.g(x...)

f(g) = x->g(x...)

# the equivalent to Base._promote_op
function _promote_op_t(F, T::ANY)
    G = Tuple{Base.Generator{Tuple{T},F}}
    Core.Inference.return_type(first, G)
end

const Zip2 = VERSION < v"0.6.0-dev" ? Base.Zip2 : Base.Iterators.Zip2

function _promote_op_t(F, R::ANY, S::ANY)
    G = Tuple{Base.Generator{Zip2{Tuple{R},Tuple{S}},_F{F}}}
    Core.Inference.return_type(first, G)
end

function _promote_op_t(F, R::ANY, S::ANY, T::ANY...)
    _promote_op_t(F, _promote_op_t(F, R, S), T...)
end

@pure function eltypes{T<:Tuple}(::Type{T})
    Tuple{map(eltype, T.parameters)...}
end

"""
`promote_arraytype{C}(::Type{F}, ::Type{C}...)`
Given a function type and types of containers with elements being mapped
figure out the output container type.
"""
@pure function promote_arraytype{F,T<:Tuple}(::Type{F}, ::Type{T})
    promote_arraytype(F, map(arraytype, T.parameters)...)
end

@pure function promote_arraytype{F}(::Type{F}, T...)
    promote_arraytype(F, promote_arraytype(F, T[1]), T[2:end]...)
end

@pure function promote_arraytype{F, T<:Array, S<:Array}(::Type{F}, ::Type{T}, ::Type{S})
    Array{_promote_op_t(F, eltype(T), eltype(S)), max(ndims(T), ndims(S))}
end

idxtype{X,I}(::Type{SparseMatrixCSC{X,I}}) = I
@pure function promote_arraytype{F, T<:SparseMatrixCSC, S<:SparseMatrixCSC}(::Type{F}, ::Type{T}, ::Type{S})
    SparseMatrixCSC{_promote_op_t(F, eltype(T), eltype(S)),
                    promote_type(idxtype(T), idxtype(S))}
end

@pure function promote_arraytype{T<:Array, S<:SparseMatrixCSC}(::Type{typeof(+)}, ::Type{T}, ::Type{S})
    Array{_promote_op_t(F, eltype(T), eltype(S))}
end

@pure function promote_arraytype{T<:Array, S<:SparseMatrixCSC}(::Type{typeof(*)}, ::Type{T}, ::Type{S})
    SparseMatrixCSC{_promote_op_t(F, eltype(T), eltype(S)),
                    promote_type(idxtype(T), idxtype(S))}
end

@pure function promote_arraytype{T<:SparseMatrixCSC, S<:Array}(::Type{typeof(+)}, ::Type{T}, ::Type{S})
    Array{_promote_op_t(F, eltype(T), eltype(S))}
end

"""
`reduce_identity(f, T::Type)`

Identity value for reducing a collection of `T` with function `f`
"""
reduce_identity{T}(f::Union{typeof(+), typeof(-)}, ::Type{T}) = zero(T)
reduce_identity{T}(f::typeof(min), ::Type{T}) = typemax(T)
reduce_identity{T}(f::typeof(max), ::Type{T}) = typemin(T)
reduce_identity{T}(f::typeof(*), ::Type{T}) = one(T)
reduce_identity{T}(f::typeof(push!), ::Type{T}) = T[]
#=

    # TODO

    - Allocate automatically!!!
    - Make sure we can generate different code for different array type combiniations
    - Use this to generate code for sparse matrices
    - Use the same to generate code for DArray

=#
