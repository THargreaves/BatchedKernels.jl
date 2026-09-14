# Selective residences, keeping the register baseline's compute variants fixed.
# OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/hybrid_selective.jl
include("hybrid_m6.jl")

function selective_assignment(tape, policy, nthreads; staging=:col)
    a = forced_assignment(tape,:register;nthreads)
    placements = Dict{Int,Int}()
    products = Int[]
    sums = Int[]
    factors = Int[]
    for (id,node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
            placements[input.index] = id
            input.index in (2,4) && (a.orientations[id] = :col)
        elseif node.fn === (*)
            push!(products,id)
        elseif node.fn === (+)
            push!(sums,id)
        elseif node.fn === cholesky
            push!(factors,id)
        end
    end
    for id in products[end-1:end]
        a.orientations[id] = :col
        a.variants[id] = :matmul_col
    end
    if policy in (:shared_H,:shared_H_predicted)
        id = placements[4]
        a.residences[id] = :single
        a.orientations[only(tape.nodes[id].args).id] = :col
    end
    policy in (:shared_predicted,:shared_H_predicted) && (a.residences[first(sums)] = :single)
    policy === :shared_factor && (a.residences[only(factors)] = :single)
    # The final product is column-oriented. Preserve it through single staging;
    # the global writer still emits the same logical column-major output.
    for (id,node) in enumerate(tape.nodes)
        if node isa BK.CallNode && BK._isstage(node.fn)
            a.orientations[id] = staging
        end
    end
    raw = [i for (i,node) in enumerate(tape.nodes) if !(node isa BK.NewNode)]
    return Assignment(tape;
        residences=Dict(i=>a.residences[i] for i in raw if haskey(a.residences,i)),
        orientations=Dict(i=>a.orientations[i] for i in raw if haskey(a.orientations,i)),
        variants=a.variants,nthreads)
end

function run_selective()
    BK.DEBUG_ACCESSORS && error("Use production preferences")
    CUDA.versioninfo()
    D = 32
    N = parse(Int,get(ENV,"BK_SELECTIVE_N","8193"))
    rng = MersenneTwister(62028)
    inputs = [Array{Float32}(undef,D,D,N) for _ in 1:5]
    P,A,Q,H,R = inputs
    reference = similar(P)
    for n in 1:N
        X = randn(rng,Float32,D,D)/sqrt(Float32(D))
        P[:,:,n] = X*X' + I
        A[:,:,n] = Matrix{Float32}(I,D,D) + .02f0*randn(rng,Float32,D,D)
        Q[:,:,n] = .2f0*Matrix{Float32}(I,D,D)
        H[:,:,n] = .1f0*randn(rng,Float32,D,D)
        R[:,:,n] = Matrix{Float32}(I,D,D)
        reference[:,:,n] = covariance_reference((x[:,:,n] for x in inputs)...)
    end
    args = Tuple(BK.BatchedCuMatrix(CuArray(x)) for x in inputs)
    tape = BK.trace(covariance_step,BK.InputSpec[BK.input_spec(x) for x in args])
    @assert BK.plan_memory(tape;order=BK.schedule(tape)).num_matrix_slots == 5
    records = []
    # Compare at the prior common geometry, then at four warps to expose whether
    # shared pressure prevents realizing a register-pressure/occupancy benefit.
    configs = if get(ENV,"BK_SELECTIVE_FINALISTS","false") == "true"
        [(threads,policy,staging) for threads in (32,128) for
            (policy,staging) in ((:register,:row),(:shared_H,:col))]
    else
        [(threads,policy,staging) for threads in (32,128) for
            (policy,staging) in ((:register,:row),(:register,:col),(:shared_H,:col),
                (:shared_predicted,:col),(:shared_factor,:col),(:shared_H_predicted,:col))]
    end
    for (threads,policy,staging) in configs
        a = selective_assignment(tape,policy,threads;staging)
        p = BK.plan_memory(tape,a;D_MAX=D)
        @assert p.num_dual_slots == 0
        @assert 1 <= p.num_single_slots <= 2
        println("PLAN,$policy,$threads,staging=$staging,single=$(p.num_single_slots),dual=$(p.num_dual_slots),elements=$(p.peak_register_elements)")
        flush(stdout)
        r = prepare(covariance_step,args,policy,reference;nthreads=threads,assignment_override=a)
        r === nothing || push!(records,merge(r,(;policy=Symbol(policy,:_,staging,:_stage))))
    end
    get(ENV,"BK_SELECTIVE_VALIDATE_ONLY","false") == "true" || measure!(records,rng)
    @assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
end
abspath(PROGRAM_FILE) == (@__FILE__) && run_selective()
