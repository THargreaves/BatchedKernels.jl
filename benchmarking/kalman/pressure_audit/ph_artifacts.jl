include("ph_inputs.jl")
reflect=parentmodule(CUDA.code_sass)
for policy in (:register,:shared_H_predicted)
    threads=128
    a=ph_assignment(tape,policy,threads;staging=:col,h_input=2)
    a=BK.Assignment(a.residences,a.orientations,a.variants,order,a.nthreads)
    entry=BK._ensure_compiled!(covariance_step_PH,args;assignment=a,nthreads=threads)
    out=similar(args[1].data);ka=(out,(x.data for x in args)...,Int32(N))
    types=Tuple{map(x->typeof(CUDA.cudaconvert(x)),ka)...}
    Base.invokelatest() do
        source=reflect.methodinstance(typeof(entry.fn),types)
        config=reflect.CUDACore.compiler_config(CUDA.device())
        job=reflect.CompilerJob(source,config)
        compiled=reflect.CUDACore.compile(job)
        write(joinpath(@__DIR__,"ph_$policy.cubin"),compiled.image)
    end
    println("SAVED,$policy");flush(stdout)
end
