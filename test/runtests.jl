using ArrayMeta
using Base.Test

@testset "utilities" begin
    @test indicesinvolved(:(A[i,j,k])) == [:A=>Any[:i,:j,:k]]
    @test indicesinvolved(:(A[i,j,k]+B[x,y,z])) == [:A=>Any[:i,:j,:k], :B=>[:x,:y,:z]]
    @test indicesinvolved(:(A[i,j,k] |> f)) == [:A=>Any[:i,:j,:k]]
end

@testset "Indexing" begin
    itr = Indexing([1], (IterSym{:i}(),))
    @test eltype(typeof(itr)) == Int64
    @test arraytype(typeof(itr)) == Array{Int64,1}
end

@testset "Map" begin
    itr1 = Indexing([1], (IterSym{:i}()))
    itr2 = Indexing([1.], (IterSym{:j}()))
    map1 = Map(*, (itr1, itr2))
    map2 = Map(/, (ConstArg(1), itr1))
    @test eltype(typeof(map1)) == Float64
    @test eltype(typeof(map2)) == Float64
    @test arraytype(typeof(map1)) == Array{Float64,1}
end

@testset "Reduce" begin
    itr1 = Indexing(rand(Int,10), (IterSym{:i}(),))
    @test eltype(Reduce(IterSym{:i}(), push!, itr1, Int[])) == Array{Int,1}
end

@testset "Lower" begin
    A = rand(2,2); B = rand(2,2); C = rand(2,2);
    i, j, k = [IterSym{x}() for x in [:i,:j,:k]]
    # map
    @test @lower(A[i,j] = B[i,j]) == ArrayOp(Indexing(A, (i, j)), Indexing(B, (i, j)))

    # transpose
    @test @lower(A[i,j] = B[j,i]) == ArrayOp(Indexing(A, (i, j)), Indexing(B, (j, i)))

    # reduced over i:
    @test @lower(A[j] = B[j,i])   == ArrayOp(Indexing(A, (j,)), Reduce(i, +, Indexing(B, (j, i))))

    # reduced over i, output is reducedim
    @test @lower(A[1,j] = B[i,j]) == ArrayOp(Indexing(A, (IterConst{Int}(1), j)), Reduce(i, +, Indexing(B, (i, j))))

    # reduce both dimensions, use * to reduce i and + to reduce j
    @test @lower(A[1,1] = B[i,j], [i=>*,j=>+]) == ArrayOp(Indexing(A, (IterConst{Int}(1), IterConst{Int}(1))),
                                                           Reduce(j, +, Reduce(i, *, Indexing(B, (i, j)))))
end

@testset "indexspaces" begin
    i,j,k=IterSym{:i}(), IterSym{:j}(), IterSym{:k}()
    itr = Indexing(rand(10,10), (i,j))
    @test (index_spaces(:X, typeof(itr))|>string ==
        Dict{Any,Any}(:i=>Any[(Array{Float64, 2}, 1, :(X.array))],
                      :j=>Any[(Array{Float64, 2}, 2, :(X.array))]) |> string)

    a = Indexing(rand(10,10), (i,k))
    b = Indexing(rand(10,10), (k,j))

   #equality problems
   #@test index_spaces(:X, typeof(Map(*, (a, b)))) ==
   #    Dict{Any, Any}(:i => Any[(Array{Float64,2}, 1, :(X.Xs[1]))],
   #                   :k => Any[(Array{Float64,2}, 2, :(X.Xs[1])),
   #                             (Array{Float64,2}, 1, :(X.arrays[2]))],
   #                   :j => Any[(Array{Float64,2}, 2, :(X.arrays[2]))],
   #    )
end

import MacroTools: striplines
@testset "kernel_expr" begin
    @testset "Indexing" begin
        X = rand(2,2);
        testtype(x) = typeof(x.rhs)
        @test kernel_expr(:X, Array{Float64, 2}, testtype(@lower(X[i,j,k] = X[i,j,k])))|>string == :(X.array[i,j,k])|>string
        @test kernel_expr(:X, Array{Float64, 2}, testtype(@lower(X[i,j,1] = X[i,j,1])))|>string == :(X.array[i,j,X.idx[3].val])|>string
    end
    @testset "Map" begin
        X = rand(2,2);
        Y = rand(2,2);
        testtype(x) = typeof(x.rhs)
        @test kernel_expr(:X,Array{Float64,2}, testtype(@lower(X[i,j,k] = -Y[i,j,k])))|>string == :(X.f(X.arrays[1].array[i,j,k]))|>string
        @test kernel_expr(:X,Array{Float64,2}, testtype(@lower(X[i,j,k] = X[i,k,j]-Y[i,j,k])))|>string == :(X.f(X.arrays[1].array[i,k,j], X.arrays[2].array[i,j,k]))|>string
    end

    @testset "Reduce" begin
        X = rand(2,2);
        Y = rand(2,2);
        testtype(x) = typeof(x.rhs)
        tex = quote
                  let tmp = start(1:size(X.array.arrays[1].array, 3))
                      if done(1:size(X.array.arrays[1].array, 3), k)
                          acc = X.empty
                      else
                          (k, tmp) = next(1:size(X.array.arrays[1].array, 3), tmp)
                          acc = X.array.f(X.array.arrays[1].array[i, j, k])
                      end
                      while !(done(1:size(X.array.arrays[1].array, 3), k))
                          (k, tmp) = next(1:size(X.array.arrays[1].array, 3), tmp)
                          acc = X.f(acc, X.array.f(X.array.arrays[1].array[i, j, k]))
                      end
                      acc
                  end
              end|>striplines

              @test kernel_expr(:X, typeof(X), testtype(@lower(X[i,j] = -Y[i,j,k])))|>striplines|>string  == string(tex)
    end
end

using Dagger
@testset "@arrayop" begin
    @testset "abstract array" begin

        A=rand(2,2); B=rand(2,2); C=rand(2,2);
        @test @arrayop(A[i,j]=B[i,k]*C[k,j]) == B*C

    end

    @testset "dagger" begin

        A = rand(Blocks(2,2), 4,4)
        B = rand(Blocks(2,2), 4,4)
        C = rand(Blocks(2,2), 4,4)

        A,B,C = map(compute, [A,B,C])
        D = map(identity, A)
        D = compute(D)
        @arrayop A[i,j] = B[i,k]*C[k,j]
        @test gather(A) ≈ gather(B)*gather(C)
    end

end
