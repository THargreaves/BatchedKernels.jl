# GPU scripts must run sequentially. Defaults to the retained-intermediate D32 case.
include("../kalman/hybrid_m6.jl")

function retained_intermediates(A,B)
    X=A*B
    Y=B*A
    U=A*A
    V=B*B
    return ((U*V)*Y)*X
end

function reused_intermediates(A,B)
    X=A*B
    Y=B*A
    U=X*X
    V=Y*Y
    return (U*V)*(X*Y)
end

function intermediate_assignment(tape,policy,threads;order=:natural)
    a=forced_assignment(tape,:register;nthreads=threads)
    products=[i for (i,n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === (*)]
    @assert length(products)==7
    shared=policy===:register ? Int[] : policy===:shared_X ? products[1:1] : products[1:2]
    for id in shared
        a.residences[id]=:single
    end
    if order===:register_min
        # Reused graph only: exact register-proxy minimum, with sequential input
        # loads to avoid pinning a second staging slot unnecessarily.
        sequence=[1,2,3,4,5,6,8,7,12,9,10,11,13,14]
        a=BK.Assignment(a.residences,a.orientations,a.variants,sequence,a.nthreads)
    end
    return a
end

function run_intermediate_storage()
    graph=Symbol(get(ENV,"MATMUL_GRAPH","retained"))
    graph in (:retained,:reused) || error("Unknown graph")
    f=graph===:retained ? retained_intermediates : reused_intermediates
    D=parse(Int,get(ENV,"MATMUL_D","32"))
    D in (16,32) || error("This benchmark targets D16 and D32")
    order=Symbol(get(ENV,"MATMUL_ORDER","natural"))
    order in (:natural,:register_min) || error("Unknown order")
    order===:register_min && graph!==:reused && error("Register-min control is for the reused graph")
    N=8193
    rng=MersenneTwister(42189)
    inputs=ntuple(_->randn(rng,Float32,D,D,N)/sqrt(Float32(D)),2)
    reference=similar(inputs[1])
    for n=1:N
        reference[:,:,n]=f(inputs[1][:,:,n],inputs[2][:,:,n])
    end
    args=Tuple(BK.BatchedCuMatrix(CuArray(x)) for x in inputs)
    tape=BK.trace(f,BK.InputSpec[BK.input_spec(x) for x in args])
    configs=order===:register_min ? [(64,:register),(64,:shared_XY)] : D==16 ?
        [(64,:register),(64,:shared_XY),(128,:register),(128,:shared_XY)] :
        [(64,:register),(64,:shared_X),(64,:shared_XY),
         (128,:register),(128,:shared_X),(128,:shared_XY)]
    records=[]
    for (threads,policy) in configs
        a=intermediate_assignment(tape,policy,threads;order)
        p=BK.plan_memory(tape,a;D_MAX=D)
        println("PLAN,$graph,$D,$order,$policy,$threads,$(p.peak_register_elements),$(p.num_single_slots),$(p.num_dual_slots)")
        r=prepare(f,args,policy,reference;nthreads=threads,assignment_override=a)
        r===nothing && error("Candidate failed numerical or zero-local-memory admission")
        push!(records,r)
    end
    for r in records;kernel_time(r);end
    for _=1:9,index in randperm(rng,length(records))
        r=records[index];push!(r.times,kernel_time(r))
    end
    for r in records
        println("TIME,$graph,$D,$order,$(r.policy),$(r.nthreads),$(r.regs),$(r.shared),$(r.occupancy),$(median(r.times)),$(quantile(r.times,.25)),$(quantile(r.times,.75))")
    end
    @assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
    println("INPUTS_UNCHANGED")
end

abspath(PROGRAM_FILE)==(@__FILE__) && run_intermediate_storage()
