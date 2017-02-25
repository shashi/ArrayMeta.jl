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
