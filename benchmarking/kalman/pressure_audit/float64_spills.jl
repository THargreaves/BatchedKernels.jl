# Focused production-resource and disassembly audit, not a throughput benchmark.
include("../float64_storage.jl")

function spill_reference(P, H, A, Q, R, s)
    predicted = A * P * A' + Q
    rhs = H * predicted
    U = cholesky(Symmetric(rhs * H' + R, :U)).U
    gain = U \ (U' \ rhs)
    correction = I - gain' * H
    return correction * predicted
end
function spill_assignment(tape, threads)
    a = forced_assignment(tape, :register; nthreads=threads)
    products = Int[]
    sums = Int[]
    for (id,node) in enumerate(tape.nodes)
        node isa BK.CallNode || continue
        if BK._isplacement(node.fn)
            input = tape.nodes[only(tape.nodes[only(node.args).id].args).id]
            if input.index == 2
                a.residences[id] = :single
                a.orientations[id] = :col
                a.orientations[only(node.args).id] = :col
            end
        elseif node.fn === (*)
            push!(products,id)
            if length(products) >= 5
                a.orientations[id] = :col
                a.variants[id] = :matmul_col
            end
        elseif node.fn === (+)
            push!(sums,id)
            length(sums) == 1 && (a.residences[id] = :single)
        elseif BK._isstage(node.fn)
            a.orientations[id] = length(products) >= 5 ? :col : :row
        end
    end
    raw = [i for (i,n) in enumerate(tape.nodes) if !(n isa BK.NewNode)]
    return Assignment(tape; residences=Dict(i=>a.residences[i] for i in raw if haskey(a.residences,i)),
        orientations=Dict(i=>a.orientations[i] for i in raw if haskey(a.orientations,i)),
        variants=a.variants,nthreads=threads)
end

function run_spill_audit()
    BK.DEBUG_ACCESSORS && error("Production accessors required")
    CUDA.versioninfo()
    D,N,threads = 32,5,parse(Int,get(ENV,"THREADS","128"))
    output_dir = get(ENV,"ARTIFACT_DIR", "/tmp/kalman_float64_spills")
    mkpath(output_dir)
    rng = MersenneTwister(42189)
    P = Array{Float64}(undef,D,D,N)
    for n in 1:N
        x = randn(rng,D,D)/sqrt(D)
        P[:,:,n] = x*x' + I
    end
    A = Matrix{Float64}(I,D,D) + 0.02randn(rng,D,D)
    Q = 0.2Matrix{Float64}(I,D,D)
    H = repeat(0.1randn(rng,D,D),1,1,N)
    R = Matrix{Float64}(I,D,D)
    inputs = (P,H,A,Q,R)
    args = (BK.BatchedCuMatrix(CuArray(P)), BK.BatchedCuMatrix(CuArray(H)),
        (BK.SharedCuMatrix(CuArray(x),N) for x in (A,Q,R))...)
    records = []
    for stage in (9,)
        f = covariance_four
        tape = BK.trace(f,BK.InputSpec[BK.input_spec(x) for x in args])
        a = spill_assignment(tape,threads)
        entry = BK._ensure_compiled!(f,args;assignment=a,nthreads=threads,shared_memory=:dynamic)
        out = similar(args[1].data)
        ka = (out,(x.data for x in args)...,Int32(N))
        kernel = Base.invokelatest() do
            fn = entry.fn
            @cuda launch=false fn(ka...)
        end
        shmem = BK._configure_dynamic_shared!(kernel,entry)
        CUDA.@sync kernel(ka...;threads,blocks=cld(N,threads÷D),shmem)
        got = Array(out)
        reference = cat((spill_reference(P[:,:,n],H[:,:,n],A,Q,R,stage) for n in 1:N)...;dims=3)
        @assert all(isapprox(got[:,:,n],reference[:,:,n];rtol=1e-11,atol=1e-12) for n in 1:N)
        mem = CUDA.memory(kernel)
        println("RESOURCE,$stage,$threads,$(CUDA.registers(kernel)),$(getproperty(mem,:local)),$(shmem+mem.shared)")
        push!(records,(stage,threads,CUDA.registers(kernel),getproperty(mem,:local),shmem+mem.shared))
        flush(stdout)
        get(ENV,"CAPTURE","true") == "true" || continue
        types = Tuple{map(x->typeof(CUDA.cudaconvert(x)),ka)...}
        reflect = parentmodule(CUDA.code_sass)
        Base.invokelatest() do
            source = reflect.methodinstance(typeof(entry.fn),types)
            job = reflect.CompilerJob(source,reflect.CUDACore.compiler_config(CUDA.device()))
            compiled = reflect.CUDACore.compile(job)
            write(joinpath(output_dir,"stage_$stage.cubin"),compiled.image)
            open(joinpath(output_dir,"stage_$stage.ptx"),"w") do io
                CUDA.code_ptx(io,entry.fn,types;kernel=true)
            end
            open(joinpath(output_dir,"stage_$stage.ll"),"w") do io
                CUDA.code_llvm(io,entry.fn,types;kernel=true)
            end
        end
    end
    @assert all(Array(x.data)==y for (x,y) in zip(args,inputs))
    open(joinpath(output_dir,"resources.csv"),"w") do io
        println(io,"stage,threads,registers,local_bytes,shared_bytes")
        foreach(r -> println(io,join(r,',')),records)
    end
    println("INPUTS_UNCHANGED")
end
abspath(PROGRAM_FILE) == (@__FILE__) && run_spill_audit()
