#=
This script confirms that if `lane` cannot be proven by the compiler to be 
within bounds, it will include a throwing branch.
It also shows that if lane can be proven to be fine, then this
throwing branch will not appear.
=#

using CUDA

function has_inexact_throws(kernel, args...)
    dev_args = map(CUDA.cudaconvert, args)
    argtypes = Tuple{map(Core.Typeof, dev_args)...}
    ir = sprint(io -> CUDA.code_llvm(io, kernel, argtypes; kernel=true))
    return count("throw_inexacterror", ir)
end

function k_throw!(out, lane)
    @inbounds out[1] = shfl_sync(0xffffffff, 1.0f0, lane, 32)
    return
end

function k_nothrow!(out, lane)
    if lane < 1 || lane > 32
        @inbounds out[1] = out[2]
    else
        @inbounds out[1] = shfl_sync(0xffffffff, 1.0f0, lane, 32)
    end
    return
end

out = CUDA.zeros(Float32, 1)
throw_counts = has_inexact_throws(
    k_throw!,
    out, Int32(5),
)
nothrow_counts = has_inexact_throws(
    k_nothrow!,
    out, Int32(5),
)
println("throw_counts=$throw_counts, nothrow_counts=$nothrow_counts")
@device_code_llvm @cuda threads=32 k_throw!(out, Int32(5))


#=
THROWING KERNEL
; PTX CompilerJob of MethodInstance for k!(::CuDeviceVector{Float32, 1}, ::Int32) for sm_89
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:3 within `k!`
define ptx_kernel void @_Z2k_13CuDeviceArrayI7Float32Li1ELi1EE5Int32({ ptr, i32 } %state, { ptr addrspace(1), i64, [1 x i64], i64 } %"out::CuDeviceArray", i32 signext %"lane::Int32") local_unnamed_addr {
conversion:
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:4 within `k!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/warp.jl:29 within `shfl_sync`
; │┌ @ int.jl:1013 within `-`
; ││┌ @ int.jl:555 within `rem`
     %0 = sext i32 %"lane::Int32" to i64  # NOTE: TO INT64
; ││└
; ││ @ int.jl:1015 within `-` @ int.jl:86
    %1 = add nsw i64 %0, -1  # NOTE: SUBTRACT 1
; │└
; │┌ @ essentials.jl:687 within `cconvert`
; ││┌ @ number.jl:7 within `convert`
; │││┌ @ boot.jl:961 within `UInt32`
; ││││┌ @ boot.jl:921 within `toUInt32`
; │││││┌ @ boot.jl:837 within `checked_trunc_uint`
        %2 = icmp ugt i64 %1, 4294967295
        br i1 %2, label %L8, label %L14  # NOTE: throwing branch if number doesn't fit into UInt

L8:                                               ; preds = %conversion
        call fastcc void @julia_throw_inexacterror_25982({ ptr, i32 } %state)
        unreachable

L14:                                              ; preds = %conversion
; ││││││ @ boot.jl:835 within `checked_trunc_uint`
        %3 = trunc i64 %1 to i32
        %"out::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [1 x i64], i64 } %"out::CuDeviceArray", 0
; │└└└└└
   %4 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float 1.000000e+00, i32 %3, i32 31)
; └
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││┌ @ none within `pointerset`
; │││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
        store float %4, ptr addrspace(1) %"out::CuDeviceArray.fca.0.extract", align 4
; └└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:5 within `k!`
  ret void
}

# throwing code:
function checked_trunc_uint(::Type{To}, x::From) where {To,From}
    @inline
    y = trunc_int(To, x)
    back = zext_int(From, y)
    eq_int(x, back) || throw_inexacterror(:trunc, To, x)  # throw
    y
end

# This is eliminated with dead code elimination if compiler has enough informaiton, e.g. max(1, lid)




NON THROWING KERNEL
; PTX CompilerJob of MethodInstance for k!(::CuDeviceVector{Float32, 1}, ::Int32) for sm_89
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:3 within `k!`
define ptx_kernel void @_Z2k_13CuDeviceArrayI7Float32Li1ELi1EE5Int32({ ptr, i32 } %state, { ptr addrspace(1), i64, [1 x i64], i64 } %"out::CuDeviceArray", i32 signext %"lane::Int32") local_unnamed_addr {
conversion:
  %"out::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [1 x i64], i64 } %"out::CuDeviceArray", 0
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:4 within `k!`
  %0 = add i32 %"lane::Int32", -1
  %or.cond = icmp ult i32 %0, 32
  br i1 %or.cond, label %L22, label %L47

L22:                                              ; preds = %conversion
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:7 within `k!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/warp.jl:29 within `shfl_sync`
   %1 = call float @llvm.nvvm.shfl.sync.idx.f32(i32 -1, float 1.000000e+00, i32 %0, i32 31)
; └
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
    br label %L67

L47:                                              ; preds = %conversion
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:5 within `k!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││┌ @ none within `pointerref`
; │││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
        %2 = getelementptr inbounds float, ptr addrspace(1) %"out::CuDeviceArray.fca.0.extract", i64 1
        %3 = load float, ptr addrspace(1) %2, align 4
; └└└└└└
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
    br label %L67

L67:                                              ; preds = %L47, %L22
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl within `k!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││┌ @ none within `pointerset`
; │││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
        %storemerge = phi float [ %3, %L47 ], [ %1, %L22 ]
        store float %storemerge, ptr addrspace(1) %"out::CuDeviceArray.fca.0.extract", align 4
; └└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/perf_engineering_verifications/shfl_sync_throwing_branch.jl:9 within `k!`
  ret void
}
=#
