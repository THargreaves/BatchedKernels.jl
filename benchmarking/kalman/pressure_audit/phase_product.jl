artifact_dir=get(ENV,"BK_PRESSURE_ARTIFACT_DIR",joinpath(@__DIR__,"../../../research/split_storage_plan/pressure_audit"))
mkpath(artifact_dir)
include("ph_inputs.jl")
product_probe(G,H)=G'*H
G=similar(P); ref=similar(P)
for n=1:N
 pred=A*P[:,:,n]*A'+Q; rhs=H[:,:,n]*pred
 U=cholesky(Symmetric(rhs*H[:,:,n]'+R,:U)).U
 G[:,:,n]=U\(U'\rhs); ref[:,:,n]=G[:,:,n]'*H[:,:,n]
end
pargs=(BK.BatchedCuMatrix(CuArray(G)),args[2])
t=BK.trace(product_probe,BK.InputSpec[BK.input_spec(x) for x in pargs])
a=forced_assignment(t,:register;nthreads=128)
for (id,node) in enumerate(t.nodes)
 node isa BK.CallNode || continue
 if BK._isplacement(node.fn)
  inp=t.nodes[only(t.nodes[only(node.args).id].args).id]
  if inp.index==2
   a.residences[id]=:single; a.orientations[id]=:col
   a.orientations[only(node.args).id]=:col
  end
 elseif node.fn==(*)
  a.orientations[id]=:col; a.variants[id]=:matmul_col
 elseif BK._isstage(node.fn)
  a.orientations[id]=:col
 end
end
raw=[i for (i,node) in enumerate(t.nodes) if !(node isa BK.NewNode)]
a=BK.Assignment(t;residences=Dict(i=>a.residences[i] for i in raw if haskey(a.residences,i)),orientations=Dict(i=>a.orientations[i] for i in raw if haskey(a.orientations,i)),variants=a.variants,nthreads=128)
r=prepare(product_probe,pargs,:product_probe,ref;assignment_override=a)
r === nothing && error("Product probe failed numerical/resource admission; see ADMISSION output")
reflect=parentmodule(CUDA.code_sass)
entry=BK._ensure_compiled!(product_probe,pargs;assignment=a,nthreads=128)
types=Tuple{map(x->typeof(CUDA.cudaconvert(x)),r.ka)...}
Base.invokelatest() do
 source=reflect.methodinstance(typeof(entry.fn),types)
 job=reflect.CompilerJob(source,reflect.CUDACore.compiler_config(CUDA.device()))
 write(joinpath(artifact_dir,"product_probe.cubin"),reflect.CUDACore.compile(job).image)
 open(joinpath(artifact_dir,"product_probe.ptx"),"w") do io
  CUDA.code_ptx(io,entry.fn,types;kernel=true)
 end
 open(joinpath(artifact_dir,"product_probe.ll"),"w") do io
  CUDA.code_llvm(io,entry.fn,types;kernel=true)
 end
end

@assert Array(pargs[1].data)==G
@assert Array(pargs[2].data)==H
