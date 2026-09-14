include("inputs.jl")
records=[]
for policy in (:legacy,:register,:shared_H)
    threads=policy===:legacy ? 32 : 128
    a=policy===:legacy ? nothing : selective_assignment(tape,policy,threads;staging=policy===:register ? :row : :col)
    entry=BK._ensure_compiled!(covariance_step,args;assignment=a,nthreads=threads)
    for cap in (policy===:legacy ? (0,) : (0,192,168,128))
        println("COMPILE,$policy,$cap");flush(stdout)
        out=similar(first(args).data)
        ka=(out,(x.data for x in args)...,Int32(N))
        kernel=Base.invokelatest() do
            fn=entry.fn
            cap==0 ? (@cuda launch=false fn(ka...)) : (@cuda launch=false maxregs=cap fn(ka...))
        end
        blocks=cld(N,threads÷32)
        CUDA.@sync kernel(ka...;threads,blocks)
        @assert isapprox(Array(out),reference;rtol=3f-4,atol=3f-5)
        mem=CUDA.memory(kernel)
        println("RESOURCE,$policy,$cap,$(CUDA.registers(kernel)),$(mem.shared),$(getproperty(mem,:local)),$(CUDA.occupancy(kernel.fun,threads))");flush(stdout)
        push!(records,(;policy,cap,kernel,ka,nthreads=threads,blocks,times=Float64[]))
    end
end
# Diagnostic only: capped spilling kernels ARE timed to test whether the cap helps.
for r in records; kernel_time(r); end
for _=1:9, index in randperm(rng,length(records))
    r=records[index];push!(r.times,kernel_time(r))
end
for r in records
    println("TIME,$(r.policy),$(r.cap),$(median(r.times)),$(quantile(r.times,.25)),$(quantile(r.times,.75))")
end
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
