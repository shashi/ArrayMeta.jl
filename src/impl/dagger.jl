using Dagger

# ugliness management
const DaggerArray = Dagger.ComputedArray

import Dagger.chunks
function chunks(arr::DaggerArray)
    chunks(arr.result)
end
# /ugliness


function onchunks(X::Indexing)
    # An iterator on the chunks
    # TODO: handle IterConsts
    let idx = X.idx
        Indexing(map(delayed(x -> Indexing(x, idx)), chunks(X.array)), X.idx)
    end
end

function onchunks(X::Map)
    let f = X.f
        Map(delayed((x...) -> Map(f, x)), map(onchunks, X.arrays))
    end
end


function onchunks{dim}(X::Reduce{dim})
    let f = X.f
        # Reduce each chunk first
        reduced_chunks = Map(delayed(x -> Reduce(dim(), f, x)), (onchunks(X.array),))

        # reduce the chunks array
        Reduce(dim(), delayed((a,b)->Map(f, (a, b))), reduced_chunks, Thunk(()->nothing)) # must be made tree reduce
    end
end

function onchunks(itr::ArrayOp)
    ArrayOp(onchunks(itr.lhs), onchunks(itr.rhs))
end

function arrayop!{D<:DaggerArray}(::Type{D}, t::ArrayOp)
    cs = arrayop!(onchunks(t))
    L(x) = Indexing(x, t.lhs.idx)
    chunksA = map(delayed(x -> arrayop!(ArrayOp(L(Array(Float64, 2,2)), x))), cs)
    t.lhs.array.result.chunks = chunksA
    t.lhs.array
end
