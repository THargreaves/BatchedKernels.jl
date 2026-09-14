include(joinpath(@__DIR__,"../hybrid_m6.jl"))
function ph_assignment(tape, policy, nthreads; staging=:col, h_input=4)
    a = BK.Assignment(tape; nthreads)
    # Match forced_assignment(tape, :register) from hybrid_m6.jl.
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            a.residences[id], a.orientations[id] = :register, :row
            a.orientations[only(node.args).id] = :row
        elseif node.fn in (*, +, -, cholesky, (\))
            a.residences[id], a.orientations[id] = :register, :row
            prefix = node.fn === (*) ? :matmul : node.fn === (+) ? :add : node.fn === (-) ? :sub : node.fn === cholesky ? :cholesky : :solve
            a.variants[id] = Symbol(prefix, :_row)
        end
    end
    placements, products, sums, factors = Dict{Int,Int}(), Int[], Int[], Int[]
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
            placements[input.index] = id
            input.index in (2, 4) && (a.orientations[id] = :col)
        elseif node.fn === (*)
            push!(products, id)
        elseif node.fn === (+)
            push!(sums, id)
        elseif node.fn === cholesky
            push!(factors, id)
        end
    end
    for id in products[end-1:end]
        a.orientations[id] = :col
        a.variants[id] = :matmul_col
    end
    if policy in (:shared_H, :shared_H_predicted)
        id = placements[h_input]
        a.residences[id] = :single
        a.orientations[only(tape.nodes[id].args).id] = :col
    end
    policy in (:shared_predicted, :shared_H_predicted) &&
        (a.residences[first(sums)] = :single)
    policy === :shared_factor && (a.residences[only(factors)] = :single)
    for (id, node) in enumerate(tape.nodes)
        node isa BK.CallNode && BK._isstage(node.fn) && (a.orientations[id] = staging)
    end
    raw = [i for (i, node) in enumerate(tape.nodes) if !(node isa BK.NewNode)]
    return BK.Assignment(tape;
        residences=Dict(i => a.residences[i] for i in raw if haskey(a.residences, i)),
        orientations=Dict(i => a.orientations[i] for i in raw if haskey(a.orientations, i)),
        variants=a.variants, nthreads)
end

covariance_step_PH(P,H,A,Q,R)=covariance_step(P,A,Q,H,R)
D,N=32,8193
rng=MersenneTwister(62028)
P=Array{Float32}(undef,D,D,N); H=similar(P)
A=Matrix{Float32}(I,D,D)+.02f0*randn(rng,Float32,D,D)
Q=.2f0*Matrix{Float32}(I,D,D); R=Matrix{Float32}(I,D,D)
reference=similar(P)
for n=1:N
    X=randn(rng,Float32,D,D)/sqrt(Float32(D))
    P[:,:,n]=X*X'+I
    H[:,:,n]=.1f0*randn(rng,Float32,D,D)
    reference[:,:,n]=covariance_reference(P[:,:,n],A,Q,H[:,:,n],R)
end
inputs=(P,H,A,Q,R)
args=(BK.BatchedCuMatrix(CuArray(P)),BK.BatchedCuMatrix(CuArray(H)),(BK.SharedCuMatrix(CuArray(x),N) for x in (A,Q,R))...)
tape=BK.trace(covariance_step_PH,BK.InputSpec[BK.input_spec(x) for x in args])
order=[4,5,1,2,3,7,10,11,12,8,13,6,14,15,16,9,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
