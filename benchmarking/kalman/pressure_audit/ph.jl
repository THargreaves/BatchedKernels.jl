include("ph_inputs.jl")
records=[]
for policy in (:legacy,:register,:shared_H_predicted)
    threads=policy===:legacy ? 32 : 128
    a=policy===:legacy ? nothing : ph_assignment(tape,policy,threads;staging=:col,h_input=2)
    if a!==nothing
        a=BK.Assignment(a.residences,a.orientations,a.variants,order,a.nthreads)
        p=BK.plan_memory(tape,a;D_MAX=D)
        println("PLAN,$policy,$(p.peak_register_elements),$(p.num_single_slots),$(p.num_dual_slots)")
    end
    entry=BK._ensure_compiled!(covariance_step_PH,args;assignment=a,nthreads=threads)
    out=similar(args[1].data); ka=(out,(x.data for x in args)...,Int32(N))
    kernel=Base.invokelatest() do
        fn=entry.fn
        @cuda launch=false fn(ka...)
    end
    blocks=cld(N,threads÷32)
    CUDA.@sync kernel(ka...;threads,blocks)
    @assert isapprox(Array(out),reference;rtol=3f-4,atol=3f-5)
    mem=CUDA.memory(kernel)
    println("RESOURCE,$policy,$(CUDA.registers(kernel)),$(mem.shared),$(getproperty(mem,:local)),$(CUDA.occupancy(kernel.fun,threads))");flush(stdout)
    push!(records,(;policy,kernel,ka,nthreads=threads,blocks,times=Float64[]))
end
# Diagnostic: retain local-memory cases to understand the controlled placement change.
for r in records; kernel_time(r); end
for _=1:9, index in randperm(rng,length(records))
    r=records[index];push!(r.times,kernel_time(r))
end
for r in records
    println("TIME,$(r.policy),$(median(r.times)),$(quantile(r.times,.25)),$(quantile(r.times,.75))")
end
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
