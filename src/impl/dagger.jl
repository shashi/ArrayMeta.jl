using Dagger
import Dagger:DomainBlocks

# ugliness management
const DArray = Dagger.ComputedArray

import Dagger.chunks
function chunks(arr::DArray)
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


function onchunks(itr::ArrayOp)
    if !hasreduceddims(itr)
        return ArrayOp(onchunks(itr.lhs), onchunks(itr.rhs))
    end
    let f = itr.reducefn
        reducefn = delayed((a,b)->Map(f, (a, b)))
        init = Thunk(()->error("Empty thunks won't be handled at the moment."))
        ArrayOp(onchunks(itr.lhs), onchunks(itr.rhs), reducefn, init)
    end
end

function arrayop!{D<:DArray}(::Type{D}, t::ArrayOp)
    cs = arrayop!(onchunks(t))
    L(x) = Indexing(x, t.lhs.idx)
    t.lhs.array.result.chunks = map(delayed((l,r) -> arrayop!(ArrayOp(L(l), r, t.reducefn, t.empty))),
                                    chunks(t.lhs.array), cs)
    t.lhs.array
end


function Base.indices(x::DArray)
    Dagger.domainchunks(x.result)
end

function Base.indices(x::DArray, i)
    idxs = indices(x)
    Dagger.DomainBlocks((idxs.start[i],), (idxs.cumlength[i],))
end

function allocarray{T,N}(::Type{DArray{T,N}}, default, idxs...)
    dmnchunks = DomainBlocks(map(i->1, idxs),
                             map(i->isa(i, DomainBlocks) ?
                                 i.cumlength[1] : (length(i),), idxs))

    chnks = map(delayed(subd -> allocarray(Array{T,1}, default, size(subd))), dmnchunks)
    sz = map((x,y)->x-y+1, map(last, dmnchunks.cumlength), dmnchunks.start)
    dmn = ArrayDomain(map(x->1:x, sz))
    DArray(Dagger.Cat(Array{T, length(idxs)}, dmn, dmnchunks, chnks))
end
