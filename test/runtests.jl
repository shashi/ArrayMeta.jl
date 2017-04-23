using ArrayMeta
using Base.Test

import ArrayMeta: indicesinvolved
@testset "utilities" begin
    @test indicesinvolved(:(A[i,j,k])) == [:A=>Any[:i,:j,:k]]
    @test indicesinvolved(:(A[i,j,k]+B[x,y,z])) == [:A=>Any[:i,:j,:k], :B=>[:x,:y,:z]]
    @test indicesinvolved(:(A[i,j,k] |> f)) == [:A=>Any[:i,:j,:k]]
end

import ArrayMeta: Indexing,Map,IndexSym,IndexConst, arraytype, ConstArg
@testset "Indexing" begin
    itr = Indexing([1], (IndexSym{:i}(),))
    @test eltype(typeof(itr)) == Int64
    @test arraytype(typeof(itr)) == Array{Int64,1}
end

@testset "Map" begin
    itr1 = Indexing([1], (IndexSym{:i}()))
    itr2 = Indexing([1.], (IndexSym{:j}()))
    map1 = Map(*, (itr1, itr2))
    map2 = Map(/, (ConstArg(1), itr1))
    @test eltype(typeof(map1)) == Float64
    @test eltype(typeof(map2)) == Float64
    @test arraytype(typeof(map1)) <: Array
end


import ArrayMeta: @lower, Assign

@testset "Lower" begin
    A = rand(2,2); B = rand(2,2); C = rand(2,2);
    i, j, k = [IndexSym{x}() for x in [:i,:j,:k]]
    # map
    @test @lower(A[i,j] = B[i,j]) == Assign(Indexing(A, (i, j)), Indexing(B, (i, j)))

    # transpose
    @test @lower(A[i,j] = B[j,i]) == Assign(Indexing(A, (i, j)), Indexing(B, (j, i)))

    # reduced over i:
    @test @lower(A[j] = B[j,i])   == Assign(Indexing(A, (j,)), Indexing(B, (j, i)))

    # reduced over i, output is reducedim
    @test @lower(A[1,j] = B[i,j]) == Assign(Indexing(A, (IndexConst{Int}(1), j)), Indexing(B, (i, j)))

    # reduce both dimensions, use * to reduce i and + to reduce j
    @test @lower(A[1,1] = B[i,j], *) == Assign(Indexing(A, (IndexConst{Int}(1), IndexConst{Int}(1))), Indexing(B, (i, j)), *, nothing)
end

import ArrayMeta: index_spaces
@testset "indexspaces" begin
    i,j,k=IndexSym{:i}(), IndexSym{:j}(), IndexSym{:k}()
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
import ArrayMeta: kernel_expr

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
end

using Dagger
import ArrayMeta: @arrayop

@testset "@arrayop" begin
    X = convert(Array, reshape(1:12, 4,3))

    Y = ones(3,4)

    # copy
    @test @arrayop(_[i,j] := X[i,j]) == X

    # transpose
    @test @arrayop(_[i,j] := X[j,i]) == X'

    # elementwise 1-arg
    @test @arrayop(_[i,j] := -X[i,j]) == -X

    # elementwise 2-args
    @test @arrayop(_[i,j] := X[i,j] + Y[j,i]) == X + Y'

    # elementwise with const
    #@test @arrayop(_[] := 2 * X[i,j])[] == sum(2.*X)

    # reduce default (+)
    @test @arrayop(_[] := X[i,j])[] == sum(X)

    # reduce with function
    @test @arrayop(_[] := X[i,j], *)[] == prod(X)

    # reducedim default (+)
    @test @arrayop(_[1, j] := X[i,j]) == sum(X, 1)
    @test @arrayop(_[i, 1] := X[i,j]) == sum(X, 2)

    # reducedim with function
    @test @arrayop(_[1, j] := X[i,j], *) == prod(X, 1)

    # broadcast
    y = [1, 2, 3, 4]
    @test @arrayop(_[i, j] := X[i, j] + y[i]) == X .+ y
    y = [1 2 3]
    @test @arrayop(_[i, j] := X[i, j] + y[1, j]) == X .+ y

    # matmul
    @test @arrayop(_[i, j] := X[i,k] * Y[k,j]) == X*Y
end

Base.:(==)(a::ArrayMeta.DArray, b::Array) = gather(a) == b

@testset "@arrayop - Dagger" begin
    X = convert(Array, reshape(1:16, 4,4))
    dX = compute(Distribute(Blocks(2,2), X))

    Y = ones(4,4)
    dY = compute(compute(Distribute(Blocks(2,2), Y))')

    # copy
    @test @arrayop(_[i,j] := dX[i,j]) == X

    # transpose
    @test @arrayop(_[i,j] := dX[j,i]) == X'

    # elementwise 1-arg
    @test @arrayop(_[i,j] := -dX[i,j]) == -X

    # elementwise 2-args
    @test @arrayop(_[i,j] := dX[i,j] + dY[j,i]) == X + Y'

    # elementwise with const
    #@test @arrayop(_[] := 2 * X[i,j])[] == sum(2.*X)

    # reduce default (+)
    @test gather(@arrayop(_[1,1] := dX[i,j])) |> first == sum(X)

    # reduce with function
    @test gather(@arrayop(_[1,1] := dX[i,j], *)) |> first == prod(X)

    # reducedim default (+)
    @test @arrayop(_[1, j] := dX[i,j]) == sum(X, 1)
    @test @arrayop(_[i, 1] := dX[i,j]) == sum(X, 2)

    # reducedim with function
    @test @arrayop(_[1, j] := dX[i,j], *) == prod(X, 1)

    # broadcast
    y = [1, 2, 3, 4]
    dy = compute(Distribute(Blocks(2), y))
    @test @arrayop(_[i, j] := dX[i, j] + dy[i]) == X .+ y
    y = [1 2 3 4]
    dy = compute(Distribute(Blocks(1,2), y))
    @test @arrayop(_[i, j] := dX[i, j] + dy[1, j]) == X .+ y

    # matmul
    @test @arrayop(_[i, j] := dX[i,k] * dY[k,j]) == X*Y
end
