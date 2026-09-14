artifact_dir=get(ENV,"BK_PRESSURE_ARTIFACT_DIR",joinpath(@__DIR__,"../../../research/split_storage_plan/pressure_audit"))
mkpath(artifact_dir)
ENV["PREFIX_DEFINITIONS_ONLY"]="true"
include("phase_prefixes.jl")
reflect=parentmodule(CUDA.code_sass)
for s in (7,8)
 @eval $(Symbol(:prefix_,s))(P,H,A,Q,R)=prefix(P,H,A,Q,R,Val($s))
 f=getfield(Main,Symbol(:prefix_,s)); t=BK.trace(f,BK.InputSpec[BK.input_spec(x) for x in args])
 a=prefix_assignment(t,:shared_H_predicted)
 entry=BK._ensure_compiled!(f,args;assignment=a,nthreads=128)
 out=similar(P);ka=(CuArray(out),(x.data for x in args)...,Int32(N))
 types=Tuple{map(x->typeof(CUDA.cudaconvert(x)),ka)...}
 Base.invokelatest() do
  source=reflect.methodinstance(typeof(entry.fn),types)
  job=reflect.CompilerJob(source,reflect.CUDACore.compiler_config(CUDA.device()))
  write(joinpath(artifact_dir,"prefix_$s.cubin"),reflect.CUDACore.compile(job).image)
  open(joinpath(artifact_dir,"prefix_$s.ptx"),"w") do io
   CUDA.code_ptx(io,entry.fn,types;kernel=true)
  end
  open(joinpath(artifact_dir,"prefix_$s.ll"),"w") do io
   CUDA.code_llvm(io,entry.fn,types;kernel=true)
  end
 end
 println("SAVED,$s");flush(stdout)
end
