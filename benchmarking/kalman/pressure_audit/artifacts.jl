include("inputs.jl")
reflect = parentmodule(CUDA.code_sass)
for policy in (:legacy,:register,:shared_H)
    threads=policy===:legacy ? 32 : 128
    a=policy===:legacy ? nothing : selective_assignment(tape,policy,threads;staging=policy===:register ? :row : :col)
    entry=BK._ensure_compiled!(covariance_step,args;assignment=a,nthreads=threads)
    out=similar(first(args).data)
    ka=(out,(x.data for x in args)...,Int32(N))
    types=Tuple{map(x->typeof(CUDA.cudaconvert(x)),ka)...}
    Base.invokelatest() do
        source=reflect.methodinstance(typeof(entry.fn),types)
        config=reflect.CUDACore.compiler_config(CUDA.device())
        job=reflect.CompilerJob(source,config)
        compiled=reflect.CUDACore.compile(job)
        write(joinpath(@__DIR__,"$policy.cubin"),compiled.image)
        open(joinpath(@__DIR__,"$policy.ptx"),"w") do io; CUDA.code_ptx(io,entry.fn,types); end
        open(joinpath(@__DIR__,"$policy.ll"),"w") do io; CUDA.code_llvm(io,entry.fn,types); end
    end
    println("SAVED,$policy");flush(stdout)
end
