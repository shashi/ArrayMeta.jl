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

@testset "Reduce meta" begin
    itr1 = Iter(rand(Int,10), (IterSym{:i}(),))
    @test eltype(Reduce(IterSym{:i}(), push!, itr1, Int[])) == Array{Int,1}
end

"""
`reduce_identity(f, T::Type)`

Identity value for reducing a collection of `T` with function `f`
"""
reduce_identity{T}(f::typeof(+), ::Type{T}) = zero(T)
reduce_identity{T}(f::typeof(*), ::Type{T}) = one(T)


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



# TODO:

"""
`optimize(t::TensorOp)`

Optimize `t` to produce an equivalent `TensorOp`
"""
function optimize(t::TensorOp)
    t
end

### Construction of loop expressions in type domain
### This is the fallback implementation for AbstractArrays

index_expr{x}(name, ::Type{IterSym{x}}, i) = x
index_expr{C<:IterConst}(name, ::Type{C}, i) = :($name.idx[$i].val)

immutable ConsState
    idx_to_dim::Dict         # used to generate size checks
    deferred_loops::Vector   # used to defer iterations to outer expression
end
ConsState() = ConsState(Dict(), [])

function inner_expr{X, Idx}(name, itr::Type{Iter{X, Idx}}, state)
    idxs = []

    for (i, idx) in  enumerate(Idx.parameters)
        j = index_expr(name, idx, i)
        push!(idxs, j)
        if isa(j, Symbol)
            # store mapping from index symbol to dimension of tensor
            Base.@get! state.idx_to_dim j []
            push!(state.idx_to_dim[j], (:($name.A), i))

            if !(j in state.deferred_loops)
                push!(state.deferred_loops, j)
            end
        end
    end

    :($name.A[$(idxs...)])
end

let
    X = rand(2,2);
    testtype(x) = typeof(x.rhs)
    state = ConsState()
    @test inner_expr(:X, testtype(@lower(X[i,j,k] = X[i,j,k])), ConsState())|>string == :(X.A[i,j,k])|>string
    @test inner_expr(:X, testtype(@lower(X[i,j,1] = X[i,j,1])), ConsState())|>string == :(X.A[i,j,X.idx[3].val])|>string
end

function inner_expr{F, Ts}(name, itr::Type{Map{F, Ts}}, state)
    innerexprs = [inner_expr(:($name.Xs[$i]), T, state) for (i, T) in enumerate(Ts.parameters)]
    :($name.f($(innerexprs...)))
end

let
    X = rand(2,2);
    Y = rand(2,2);
    testtype(x) = typeof(x.rhs)
    state = ConsState()
    @test inner_expr(:X, testtype(@lower(X[i,j,k] = -Y[i,j,k])), ConsState())|>string == :(X.f(X.Xs[1].A[i,j,k]))|>string
    @test inner_expr(:X, testtype(@lower(X[i,j,k] = X[i,k,j]-Y[i,j,k])), ConsState())|>string == :(X.f(X.Xs[1].A[i,k,j], X.Xs[2].A[i,j,k]))|>string
end

function inner_expr{idx, F, T, E}(name, itr::Type{Reduce{IterSym{idx}, F, T, E}}, state)
    inner = inner_expr(:($name.X), T, state)
    !haskey(state.idx_to_dim, idx) && throw(ArgumentError("Reduced dimension $idx unknown"))
    dim = first(state.idx_to_dim[idx])
    quote
        let $idx = 1, acc = $inner
            for $idx = 2:size($(dim...))
                acc = $name.f(acc, $inner)
            end
            acc
        end
    end
end

import MacroTools: striplines
let
    X = rand(2,2);
    Y = rand(2,2);
    testtype(x) = typeof(x.rhs)
    state = ConsState()
    tex = quote
            let k = 1, acc = X.X.f(X.X.Xs[1].A[i,j,k])
                  for k = 2:size(X.X.Xs[1].A,3)
                      acc = X.f(acc,X.X.f(X.X.Xs[1].A[i,j,k]))
                  end
                  acc
              end
          end|>striplines

    @test inner_expr(:X, testtype(@lower(X[i,j] = -Y[i,j,k])), ConsState())|>striplines|>string  == string(tex)
end

allequal(x) = true
function allequal(x, xs...)
    x == xs[1] && allequal(xs...)
end


function tensorop_body{L,R}(name, top::Type{TensorOp{L,R}})
    state = ConsState()
    rhs_inner = inner_expr(:($name.rhs), R, state)
    lhs_inner = inner_expr(:($name.lhs), L, state)

    # wrap with loops
    expr = :($lhs_inner = $rhs_inner)
    checks = :()
    for sym in reverse(state.deferred_loops)
        if !haskey(state.idx_to_dim, sym)
            throw(ArgumentError("Unknown dimension $sym"))
        end
        dims = state.idx_to_dim[sym]
        if length(dims) > 1
            # check dimensions for equality
            equal_dims = [:(size($(d...))) for d in dims]
            checks = :($checks; @assert allequal($(equal_dims...),))
        end
        dim = first(dims)
        expr = quote
            for $sym = 1:size($(dim...))
                $expr
            end
        end
    end
    :($checks; $expr; $name.lhs.A)
end

@inline @generated function top!(t::TensorOp)
    tensorop_body(:t, t)
end

"""
`top!(t::TensorOp)`

Perform a tensor operation
"""
macro top(expr, reductions=:nothing)
    :(top!(@lower $expr $reductions))
end

let
    A=rand(2,2); B=rand(2,2); C=rand(2,2);
    @test @top(A[i,j]=B[i,k]*C[k,j]) == B*C
end
