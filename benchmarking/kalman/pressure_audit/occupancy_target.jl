# D32 all-five-batched occupancy experiment; run GPU scripts sequentially.
# Default: five selected placement/orientation/resource candidates.
# BK_OCCUPANCY_THRESHOLD_ONLY=true: paired uncapped/cap168 diagnostic.
# Capped cases deliberately permit local memory and are not production candidates.
include("inputs.jl")
for name in (:MAX_SHARED_MEMORY_PER_MULTIPROCESSOR,:RESERVED_SHARED_MEMORY_PER_BLOCK,:MAX_REGISTERS_PER_MULTIPROCESSOR,:MAX_THREADS_PER_MULTIPROCESSOR)
    println("DEVICE,$name,$(CUDA.attribute(CUDA.device(),getproperty(CUDA,Symbol(:DEVICE_ATTRIBUTE_,name))))")
end
configs=get(ENV,"BK_OCCUPANCY_THRESHOLD_ONLY","false")=="true" ? [(:shared_H,:col,:col,0),(:shared_H,:col,:col,168)] : [(:register,:row,:col,0),(:shared_H,:col,:col,0),(:shared_H_predicted,:col,:col,0),(:dual_H,:row,:bothrow,0),(:shared_H,:col,:col,200)]
records=[]
for (policy,staging,correction,cap) in configs
    threads=64
    a=selective_assignment(tape,policy===:dual_H ? :shared_H : policy,threads;staging)
    if policy===:dual_H
        for (id,node) in enumerate(tape.nodes)
            node isa BK.CallNode && BK._isplacement(node.fn) || continue
            input=tape.nodes[only(tape.nodes[only(node.args).id].args).id]
            if input.index==4;a.residences[id]=:dual;a.orientations[id]=:both;end
        end
    end
    products=[i for (i,n) in enumerate(tape.nodes) if n isa BK.CallNode && n.fn === (*)]
    if correction in (:row,:bothrow)
        for id in (correction===:row ? products[end-1:end-1] : products[end-1:end])
            a.orientations[id]=:row; a.variants[id]=:matmul_row
        end
        raw=[i for (i,n) in enumerate(tape.nodes) if !(n isa BK.NewNode)]
        a=BK.Assignment(tape;residences=Dict(i=>a.residences[i] for i in raw if haskey(a.residences,i)),orientations=Dict(i=>a.orientations[i] for i in raw if haskey(a.orientations,i)),variants=a.variants,nthreads=threads)
    end
    # Validate access/orientation requirements before GPU compilation.
    p=BK.plan_memory(tape,a;D_MAX=D)
    name=Symbol(policy,:_,staging,:_,correction,:_,cap)
    println("PREPARE,$name");flush(stdout)
    entry=BK._ensure_compiled!(covariance_step,args;assignment=a,nthreads=threads)
    out=similar(args[1].data);ka=(out,(x.data for x in args)...,Int32(N))
    kernel=Base.invokelatest() do
        fn=entry.fn
        cap==0 ? (@cuda launch=false fn(ka...)) : (@cuda launch=false maxregs=cap fn(ka...))
    end
    blocks=cld(N,threads÷32)
    CUDA.@sync kernel(ka...;threads,blocks)
    @assert isapprox(Array(out),reference;rtol=3f-4,atol=3f-5)
    mem=CUDA.memory(kernel)
    cap == 0 && getproperty(mem,:local) != 0 && error("Uncapped candidate uses local memory; do not admit it as a spill-free candidate")
    println("RESOURCE,$name,$(CUDA.registers(kernel)),$(mem.shared),$(getproperty(mem,:local)),$(CUDA.active_blocks(kernel.fun,threads)),$(CUDA.occupancy(kernel.fun,threads)),$(p.num_single_slots),$(p.num_dual_slots),$(p.peak_register_elements)");flush(stdout)
    push!(records,(;policy=name,kernel,ka,nthreads=threads,blocks,times=Float64[]))
end
for r in records;kernel_time(r);end
for _=1:9,index in randperm(rng,length(records))
    r=records[index];push!(r.times,kernel_time(r))
end
for r in records;println("TIME,$(r.policy),$(median(r.times)),$(quantile(r.times,.25)),$(quantile(r.times,.75))");end
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
println("INPUTS_UNCHANGED")
