# Research-only method override in this Julia process; no source files are edited.
include("ph_inputs.jl")
source=read(joinpath(@__DIR__,"../../../src/variant_elementwise.jl"),String)
start=findfirst("@inline function variant_op!",source).start
stop=findnext("@inline function variant_op!",source,start+1).start-1
body=source[start:stop]
mode=get(ENV,"BK_PHASE_ABLATION","guards")
if mode=="guards"
    @assert count("d <= Int32(P)",body)==2
    body=replace(body,"d <= Int32(P)"=>"(P == D || d <= Int32(P))")
elseif mode=="row_fence"
    # Diagnostic only: all active D=32 groups are complete warps here.
    @assert count("ours_write!(C, i, d, accumulator, RowAccess())",body)==1
    body=replace(body,"        end\n    end\n    return C"=>"        end\n        sync_warp()\n    end\n    return C")
    @assert count("sync_warp()",body)==1
else
    error("Unknown ablation: $mode")
end
Base.include_string(BK,body,"phase_ablation_override.jl")
records=[]
for policy in (:register,:shared_H_predicted)
 a=ph_assignment(tape,policy,128;staging=:col,h_input=2)
 a=BK.Assignment(a.residences,a.orientations,a.variants,order,a.nthreads)
 r=prepare(covariance_step_PH,args,Symbol(policy,:_,mode),reference;assignment_override=a)
 r===nothing || push!(records,r)
end
measure!(records,rng)
@assert all(Array(x.data)==original for (x,original) in zip(args,inputs))
