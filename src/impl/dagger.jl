### Construction of loop expressions in type domain
### This is the fallback implementation for AbstractArrays


### TODO: dispatch to choose back-end. For now, split this into dtensorop

function onchunks(X::Iter)
    # An iterator on the chunks of iterators
    # TODO: handle IterConsts
    let idx = X.idx
        Iter(map(c -> Thunk(x -> Iter(x, idx), c), chunks(X.A)), X.idx)
    end
end

function onchunks(X::Map)
    let f = X.f
        Map((args...) -> Thunk((x...) -> Map(f, x), args...),
            map(onchunks, X.Xs))
    end
end


function onchunks{dim}(X::Reduce{dim})
    let f = X.f
        # Reduce each chunk first
        reduced_chunks = Map(c -> Thunk(x -> Reduce(dim(), f, x), c), (onchunks(X.X),))

        # reduce the chunks array
        Reduce(dim(), (x,y) -> Thunk((a,b)->Map(f, (a, b)), x, y),
            reduced_chunks, Thunk(()->nothing)) # must be made tree reduce
    end
end

function onchunks(itr::TensorOp)
    TensorOp(onchunks(itr.lhs), onchunks(itr.rhs))
end

function dtop!(t::TensorOp)
    cs = top!(onchunks(t))
    L(x) = Iter(x, t.lhs.idx)
    chunksA = map(c -> Thunk(x -> top!(TensorOp(L(Array(Float64, 2,2)), x)), c), cs)
    t.lhs.A.result.chunks = chunksA
    t.lhs.A
end

macro dtop(expr, reductions=:nothing)
    :(dtop!(@lower $expr $reductions))
end

using Dagger
import Dagger.chunks

function chunks(arr::Dagger.ComputedArray)
    chunks(arr.result)
end

let
    A = rand(Blocks(2,2), 4,4); B = rand(Blocks(2,2), 4,4); C = rand(Blocks(2,2), 4,4)
    A,B,C = map(compute, [A,B,C])
    D = map(identity, A)
    D = compute(D)

    @dtop A[i,j] = B[i,k]*C[k,j]
    @test gather(A) ≈ gather(B)*gather(C)
end
