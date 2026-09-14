# Focused five-live-slot pressure case; no search-space sweep.
# OPENBLAS_NUM_THREADS=1 julia --project=. benchmarking/kalman/hybrid_five_slots.jl
include("hybrid_m6.jl")
BK.DEBUG_ACCESSORS && error("Use production accessor preferences")
CUDA.versioninfo()
D, N = 32, 8193
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
legacy_plan = BK.plan_memory(tape;order=BK.schedule(tape))
@assert legacy_plan.num_matrix_slots == 5
println("Scheduled legacy matrix slots: ",legacy_plan.num_matrix_slots)
records = []
for policy in (:legacy,:register,:shared_inputs_register_results)
    assignment = nothing
    if policy !== :legacy
        assignment = forced_assignment(tape,:register;nthreads=32)
        for (id,node) in enumerate(tape.nodes)
            if node isa BK.CallNode && BK._isplacement(node.fn)
                input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
                if policy === :shared_inputs_register_results
                    assignment.residences[id] = :dual
                    assignment.orientations[id] = :both
                elseif input.index in (2,4) # A and H are read through adjoints too.
                    assignment.orientations[id] = :col
                end
            end
        end
        if policy === :register
            products = [i for (i,node) in enumerate(tape.nodes) if node isa BK.CallNode && node.fn === (*)]
            for id in products[end-1:end]
                assignment.orientations[id] = :col
                assignment.variants[id] = :matmul_col
            end
        end
        # Rebuild wrapper metadata after changing parent residence/orientation.
        raw = [i for (i,node) in enumerate(tape.nodes) if !(node isa BK.NewNode)]
        assignment = Assignment(tape;
            residences=Dict(i=>assignment.residences[i] for i in raw if haskey(assignment.residences,i)),
            orientations=Dict(i=>assignment.orientations[i] for i in raw if haskey(assignment.orientations,i)),
            variants=assignment.variants,nthreads=32)
    end
    record = prepare(covariance_step,args,policy,reference;nthreads=32,assignment_override=assignment)
    record === nothing || push!(records,record)
end
measure!(records,rng)
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
