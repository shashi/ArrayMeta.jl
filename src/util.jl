import Base: @pure

flatten(xs) = reduce(vcat, vec.(xs))

function indicesinvolved(expr)
    @match expr begin
        A_[idx__] => [(A => idx)]
        f_(ex__)  => reduce(vcat, [indicesinvolved(x) for x in ex])
        _ => []
    end
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
Type Domain `promote_op`

The first argument is the type of the function
"""
@pure @generated function promote_op_t{F,T}(::Type{F}, ::Type{T})
    :(_promote_op_t(F, $(T.parameters...)))
end

"""
`promote_arraytype(F::Type, Ts::Type...)`

Returns an output array type for the result of applying a function of type `F`
on arrays of type `Ts`.
"""

@pure function promote_arraytype{F}(::Type{F}, T...)
    length(T) == 1 && return T[1]
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
`reduction_identity(f, T::Type)`

Identity value for reducing a collection of `T` with function `f`
"""
reduction_identity{T}(f::Union{typeof(+), typeof(-)}, ::Type{T}) = zero(T)
reduction_identity{T}(f::typeof(min), ::Type{T}) = typemax(T)
reduction_identity{T}(f::typeof(max), ::Type{T}) = typemin(T)
reduction_identity{T}(f::typeof(*), ::Type{T}) = one(T)
reduction_identity{T}(f::typeof(push!), ::Type{T}) = T[]

function merge_dictofvecs(dicts...)
    merged_dict = Dict()
    for dict in dicts
        for (k, v) in dict
            if k in keys(merged_dict)
                merged_dict[k] = vcat(merged_dict[k], v)
            else
                merged_dict[k] = v
            end
        end
    end
    merged_dict
end
