; PTX CompilerJob of MethodInstance for kernel_mul_defrag!(::CuDeviceArray{Float32, 3, 1}, ::CuDeviceArray{Float32, 3, 1}, ::CuDeviceMatrix{Float32, 1}, ::Val{4}, ::Val{8}, ::Val{8}, ::Val{256}, ::Val{10}, ::Int32) for sm_89
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:44 within `kernel_mul_defrag!`
define ptx_kernel void @_Z18kernel_mul_defrag_13CuDeviceArrayI7Float32Li3ELi1EES1_S_IS0_Li2ELi1EE3ValILi4EES3_ILi8EES5_S3_ILi256EES3_ILi10EE5Int32({ ptr, i32 } %state, { ptr addrspace(1), i64, [3 x i64], i64 } %"M_out::CuDeviceArray", { ptr addrspace(1), i64, [3 x i64], i64 } %"M_in::CuDeviceArray", { ptr addrspace(1), i64, [2 x i64], i64 } %"A_global::CuDeviceArray", i32 signext %"N::Int32") local_unnamed_addr {
conversion:
  %"M_out::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [3 x i64], i64 } %"M_out::CuDeviceArray", 0
  %"M_in::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [3 x i64], i64 } %"M_in::CuDeviceArray", 0
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:55 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `_index`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
      %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:63 within `kernel_mul_defrag!`
; ┌ @ promotion.jl:637 within `==`
   %1 = icmp ugt i32 %0, 31
; └
  br i1 %1, label %conversion.pass44_crit_edge, label %pass13

conversion.pass44_crit_edge:                      ; preds = %conversion
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:69 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:26 within `get_shmem_elems`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; │││┌ @ int.jl:87 within `+`
      %.pre = add nuw nsw i32 %0, 1
; │└└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %.pre1 = and i32 %.pre, 31
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:63 within `kernel_mul_defrag!`
  br label %pass44

L287:                                             ; preds = %pass114, %pass44
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:76 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1191 within `intermediate_layout_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %2 = add nuw nsw i32 %84, 31
     %3 = add nuw nsw i32 %84, 30
     %4 = lshr i32 %3, 5
     %.zext63 = and i32 %4, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %5 = add i32 %85, %.zext63
; ││└
; ││┌ @ int.jl:520 within `<=`
     %6 = icmp ugt i32 %84, 993
     %.not33.1 = icmp sgt i32 %5, %"N::Int32"
; ││└
    %or.cond.1 = select i1 %6, i1 true, i1 %.not33.1
    %.not34.1 = icmp ugt i32 %2, %86
    %or.cond96 = select i1 %or.cond.1, i1 true, i1 %.not34.1
    br i1 %or.cond96, label %L287.1, label %pass114.1

pass114.1:                                        ; preds = %L287
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %7 = sext i32 %89 to i64
          %8 = getelementptr float, ptr addrspace(3) @shmem60, i64 %7
          %9 = getelementptr float, ptr addrspace(3) %8, i64 33
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %10 = add i32 %88, %2
; ││││││││└
          %11 = sext i32 %10 to i64
          %12 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %11
          %13 = load float, ptr addrspace(1) %12, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %13, ptr addrspace(3) %9, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L287.1

L287.1:                                           ; preds = %pass114.1, %L287
; ││└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %14 = add nuw nsw i32 %84, 63
     %15 = add nuw nsw i32 %84, 62
     %16 = lshr i32 %15, 5
     %.zext65 = and i32 %16, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %17 = add i32 %85, %.zext65
; ││└
; ││┌ @ int.jl:520 within `<=`
     %18 = icmp ugt i32 %84, 961
     %.not33.2 = icmp sgt i32 %17, %"N::Int32"
; ││└
    %or.cond.2 = select i1 %18, i1 true, i1 %.not33.2
    %.not34.2 = icmp ugt i32 %14, %86
    %or.cond97 = select i1 %or.cond.2, i1 true, i1 %.not34.2
    br i1 %or.cond97, label %L287.2, label %pass114.2

pass114.2:                                        ; preds = %L287.1
    %.lhs.trunc93 = add nuw nsw i32 %69, 63
    %.zext94 = lshr i32 %.lhs.trunc93, 5
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %19 = add i32 %88, %14
; ││││││││└
          %20 = sext i32 %19 to i64
          %21 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %20
          %22 = load float, ptr addrspace(1) %21, align 4
; ││└└└└└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1205
; ││┌ @ int.jl:87 within `+`
     %23 = add nuw nsw i32 %89, 64
; ││└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %24 = add nuw nsw i32 %23, %.zext94
           %25 = zext nneg i32 %24 to i64
; ││││││││└
          %26 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %25
          store float %22, ptr addrspace(3) %26, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L287.2

L287.2:                                           ; preds = %pass114.2, %L287.1
; ││└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %27 = add nuw nsw i32 %84, 95
     %28 = add nuw nsw i32 %84, 94
     %29 = lshr i32 %28, 5
     %.zext67 = and i32 %29, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %30 = add i32 %85, %.zext67
; ││└
; ││┌ @ int.jl:520 within `<=`
     %31 = icmp ugt i32 %84, 929
     %.not33.3 = icmp sgt i32 %30, %"N::Int32"
; ││└
    %or.cond.3 = select i1 %31, i1 true, i1 %.not33.3
    %.not34.3 = icmp ugt i32 %27, %86
    %or.cond98 = select i1 %or.cond.3, i1 true, i1 %.not34.3
    br i1 %or.cond98, label %pass146, label %pass114.3

pass114.3:                                        ; preds = %L287.2
    %.lhs.trunc91 = add nuw nsw i32 %69, 95
    %.zext92 = lshr i32 %.lhs.trunc91, 5
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %32 = add i32 %88, %27
; ││││││││└
          %33 = sext i32 %32 to i64
          %34 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %33
          %35 = load float, ptr addrspace(1) %34, align 4
; ││└└└└└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1205
; ││┌ @ int.jl:87 within `+`
     %36 = add nuw nsw i32 %89, 96
; ││└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %37 = add nuw nsw i32 %36, %.zext92
           %38 = zext nneg i32 %37 to i64
; ││││││││└
          %39 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %38
          store float %35, ptr addrspace(3) %39, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %pass146

L432:                                             ; preds = %L365.preheader, %pass146
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:79 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   call void @llvm.nvvm.barrier0()
   %.not40 = icmp sgt i32 %79, %"N::Int32"
   %40 = icmp ugt i32 %0, 127
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:82 within `kernel_mul_defrag!`
  %or.cond9 = or i1 %40, %.not40
  %41 = icmp ugt i32 %74, 31
  %or.cond11 = or i1 %41, %or.cond9
  br i1 %or.cond11, label %L866, label %pass171

L866:                                             ; preds = %pass171, %L432
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:95 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   call void @llvm.nvvm.barrier0()
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:98 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1360 within `dual_to_interm_transfer!`
; │┌ @ int.jl:87 within `+`
    %42 = add i32 %85, %105
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1361 within `dual_to_interm_transfer!`
; │┌ @ int.jl:87 within `+`
    %43 = add i32 %42, %.zext55
    %.not46 = icmp sgt i32 %43, %"N::Int32"
    %44 = icmp ugt i32 %104, 4
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1363 within `dual_to_interm_transfer!`
   %or.cond15 = select i1 %.not46, i1 true, i1 %44
   br i1 %or.cond15, label %pass248, label %L946.preheader

L1109:                                            ; preds = %pass259, %pass248
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:99 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1429 within `intermediate_layout_write!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %45 = add nuw nsw i32 %1257, 31
     %46 = add nuw nsw i32 %1257, 30
     %47 = lshr i32 %46, 4
     %.zext71 = and i32 %47, 4095
; ││└
; ││┌ @ int.jl:87 within `+`
     %48 = add i32 %85, %.zext71
; ││└
; ││┌ @ int.jl:520 within `<=`
     %49 = icmp ugt i32 %1257, 481
     %.not49.1 = icmp sgt i32 %48, %"N::Int32"
; ││└
    %or.cond17.1 = select i1 %49, i1 true, i1 %.not49.1
    %.not50.1 = icmp ugt i32 %45, %1258
    %or.cond100 = select i1 %or.cond17.1, i1 true, i1 %.not50.1
    br i1 %or.cond100, label %L1109.1, label %pass259.1

pass259.1:                                        ; preds = %L1109
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1445
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %50 = add i32 %1260, %45
; ││││││││└
          %51 = sext i32 %50 to i64
          %52 = getelementptr inbounds float, ptr addrspace(1) %"M_out::CuDeviceArray.fca.0.extract", i64 %51
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %53 = sext i32 %89 to i64
          %54 = getelementptr float, ptr addrspace(3) @shmem57, i64 %53
          %55 = getelementptr float, ptr addrspace(3) %54, i64 33
          %56 = load float, ptr addrspace(3) %55, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %56, ptr addrspace(1) %52, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L1109.1

L1109.1:                                          ; preds = %pass259.1, %L1109
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:101 within `kernel_mul_defrag!`
  ret void

pass13:                                           ; preds = %conversion
  %"A_global::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [2 x i64], i64 } %"A_global::CuDeviceArray", 0
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:64 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:625 within `shared_matrix_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; │││┌ @ int.jl:87 within `+`
      %57 = add nuw nsw i32 %0, 1
; │└└└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:626 within `shared_matrix_load!`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %58 = and i32 %57, 31
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not = icmp eq i32 %58, 0
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:634 within `shared_matrix_load!`
; │┌ @ int.jl:86 within `-`
    %59 = add nsw i32 %58, -1
    %60 = select i1 %.not, i32 31, i32 %59
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:639 within `shared_matrix_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││┌ @ none within `pointerref`
; ││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
         %61 = zext nneg i32 %60 to i64
         %62 = getelementptr inbounds float, ptr addrspace(1) %"A_global::CuDeviceArray.fca.0.extract", i64 %61
         %63 = load float, ptr addrspace(1) %62, align 4
; │└└└└└└
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││┌ @ none within `pointerset`
; ││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
         %64 = getelementptr inbounds float, ptr addrspace(3) @shmem, i64 %61
         store float %63, ptr addrspace(3) %64, align 4
; └└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:69 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ div.jl:336 within `fld`
; ││││┌ @ div.jl:370 within `div` @ div.jl:325 @ int.jl:301
       br label %pass44

pass44:                                           ; preds = %pass13, %conversion.pass44_crit_edge
; │││└└
; │││┌ @ int.jl:86 within `-`
      %.pre-phi2 = phi i32 [ %.pre1, %conversion.pass44_crit_edge ], [ %58, %pass13 ]
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not29 = icmp eq i32 %.pre-phi2, 0
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:30 within `get_shmem_elems`
; │┌ @ int.jl:301 within `div`
    %.lhs.trunc = add nuw nsw i32 %.pre-phi2, 255
    %65 = lshr i32 %.lhs.trunc, 3
    %.zext = and i32 %65, 31
; │└
; │┌ @ int.jl:87 within `+`
    %66 = add nuw nsw i32 %.zext, 1
    %67 = select i1 %.not29, i32 4, i32 %66
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:70 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:27 within `get_shmem_elems`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:85 within `blockIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:56 within `blockIdx_x`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `_index`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
       %68 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; │└└└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ essentials.jl:799 within `ifelse`
     %69 = select i1 %.not29, i32 32, i32 %.pre-phi2
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:30 within `get_shmem_elems`
; │┌ @ int.jl:86 within `-`
    %70 = add nsw i32 %69, -1
; │└
; │┌ @ int.jl:301 within `div`
    %71 = lshr i32 %70, 2
    %.zext53 = and i32 %71, 63
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:31 within `get_shmem_elems`
; │┌ @ int.jl:88 within `*`
    %72 = lshr i32 %0, 2
    %73 = and i32 %72, 248
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:30 within `get_shmem_elems`
; │┌ @ int.jl:87 within `+`
    %74 = add nuw nsw i32 %.zext53, %73
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:31 within `get_shmem_elems`
; │┌ @ int.jl:87 within `+`
    %75 = add nuw nsw i32 %74, 1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:32 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %76 = and i32 %69, 3
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not31 = icmp eq i32 %76, 0
; ││└
; ││┌ @ essentials.jl:799 within `ifelse`
     %77 = select i1 %.not31, i32 4, i32 %76
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:36 within `get_shmem_elems`
; │┌ @ int.jl:88 within `*`
    %78 = shl i32 %68, 5
; │└
; │┌ @ int.jl:87 within `+`
    %79 = add i32 %75, %78
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:76 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1179 within `intermediate_layout_load!`
; │┌ @ int.jl:301 within `div`
    %80 = lshr i32 %0, 5
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1186 within `intermediate_layout_load!`
; │┌ @ int.jl:88 within `*`
    %81 = shl nuw nsw i32 %80, 7
; │└
; │┌ @ int.jl:87 within `+`
    %82 = or disjoint i32 %81, 1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1187 within `intermediate_layout_load!`
; │┌ @ int.jl:88 within `*`
    %83 = mul nuw nsw i32 %80, 284
    %84 = add nuw nsw i32 %82, %69
    %85 = or disjoint i32 %78, 1
    %86 = add nuw nsw i32 %81, 128
    %87 = shl i32 %68, 10
    %88 = add i32 %87, -1
    %89 = add nuw nsw i32 %70, %83
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1191 within `intermediate_layout_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %90 = add nsw i32 %84, -1
     %91 = add nsw i32 %84, -2
; ││└
; ││┌ @ int.jl:301 within `div`
     %92 = lshr i32 %91, 5
     %.zext61 = and i32 %92, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %93 = add i32 %85, %.zext61
; ││└
; ││┌ @ int.jl:520 within `<=`
     %94 = icmp ugt i32 %91, 1023
     %.not33 = icmp sgt i32 %93, %"N::Int32"
; ││└
    %or.cond = select i1 %94, i1 true, i1 %.not33
    %.not34 = icmp ugt i32 %90, %86
    %or.cond95 = select i1 %or.cond, i1 true, i1 %.not34
    br i1 %or.cond95, label %L287, label %pass114

pass114:                                          ; preds = %pass44
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %95 = zext nneg i32 %89 to i64
          %96 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %95
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %97 = add i32 %88, %90
; ││││││││└
          %98 = sext i32 %97 to i64
          %99 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %98
          %100 = load float, ptr addrspace(1) %99, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %100, ptr addrspace(3) %96, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L287

pass146:                                          ; preds = %pass114.3, %L287.2
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:77 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1278 within `interm_to_dual_transfer!`
; │┌ @ int.jl:301 within `div`
    %101 = lshr i32 %70, 3
    %.zext55 = and i32 %101, 31
; │└
; │┌ @ int.jl:87 within `+`
    %102 = add nuw nsw i32 %.zext55, 1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1279 within `interm_to_dual_transfer!`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %103 = and i32 %69, 7
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not37 = icmp eq i32 %103, 0
; ││└
; ││┌ @ essentials.jl:799 within `ifelse`
     %104 = select i1 %.not37, i32 8, i32 %103
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1280 within `interm_to_dual_transfer!`
; │┌ @ int.jl:88 within `*`
    %105 = shl nuw nsw i32 %80, 2
; │└
; │┌ @ int.jl:87 within `+`
    %106 = add i32 %105, %78
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1281 within `interm_to_dual_transfer!`
; │┌ @ int.jl:87 within `+`
    %107 = add i32 %106, %102
    %.not38 = icmp sgt i32 %107, %"N::Int32"
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1283 within `interm_to_dual_transfer!`
   br i1 %.not38, label %L432, label %L365.preheader

L365.preheader:                                   ; preds = %pass146
   %108 = shl nuw nsw i32 %.zext55, 5
   %109 = shl nuw nsw i32 %104, 2
   %110 = add nsw i32 %109, -4
   %111 = add nsw i32 %110, %108
   %112 = shl nuw nsw i32 %80, 3
   %113 = add nsw i32 %112, -1
   %114 = add nsw i32 %113, %104
   %115 = mul nsw i32 %114, 36
   %116 = sub nsw i32 %.zext55, %105
   %117 = add nsw i32 %116, -4
   %118 = add nsw i32 %117, %115
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1284 within `interm_to_dual_transfer!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc83 = trunc i32 %111 to i8
     %119 = sdiv i8 %.lhs.trunc83, 32
     %.sext84 = sext i8 %119 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %120 = add nsw i32 %111, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %121 = add nsw i32 %120, %.sext84
; ││││││││└
          %122 = sext i32 %121 to i64
          %123 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %122
          %124 = load float, ptr addrspace(3) %123, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %125 = add nsw i32 %116, %115
; ││││││││└
          %126 = sext i32 %125 to i64
          %127 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %126
          store float %124, ptr addrspace(3) %127, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %128 = or disjoint i32 %111, 1
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc85 = trunc i32 %128 to i8
     %129 = sdiv i8 %.lhs.trunc85, 32
     %.sext86 = sext i8 %129 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %130 = add nsw i32 %128, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %131 = add nsw i32 %130, %.sext86
; ││││││││└
          %132 = sext i32 %131 to i64
          %133 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %132
          %134 = load float, ptr addrspace(3) %133, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %135 = sext i32 %118 to i64
          %136 = getelementptr float, ptr addrspace(3) @shmem57, i64 %135
          %137 = getelementptr float, ptr addrspace(3) %136, i64 8
          store float %134, ptr addrspace(3) %137, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %138 = or disjoint i32 %111, 2
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc87 = trunc i32 %138 to i8
     %139 = sdiv i8 %.lhs.trunc87, 32
     %.sext88 = sext i8 %139 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %140 = add nsw i32 %138, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %141 = add nsw i32 %140, %.sext88
; ││││││││└
          %142 = sext i32 %141 to i64
          %143 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %142
          %144 = load float, ptr addrspace(3) %143, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %145 = getelementptr float, ptr addrspace(3) %136, i64 12
          store float %144, ptr addrspace(3) %145, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %146 = or disjoint i32 %111, 3
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc89 = trunc i32 %146 to i8
     %147 = sdiv i8 %.lhs.trunc89, 32
     %.sext90 = sext i8 %147 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %148 = add nsw i32 %146, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %149 = add nsw i32 %148, %.sext90
; ││││││││└
          %150 = sext i32 %149 to i64
          %151 = getelementptr inbounds float, ptr addrspace(3) @shmem60, i64 %150
          %152 = load float, ptr addrspace(3) %151, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %153 = getelementptr float, ptr addrspace(3) %136, i64 16
          store float %152, ptr addrspace(3) %153, align 4
; └└└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:79 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   br label %L432

pass171:                                          ; preds = %L432
   %154 = lshr i32 %74, 2
   %.zext80 = and i32 %154, 63
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:84 within `kernel_mul_defrag!`
; ┌ @ operators.jl:885 within `mod1`
; │┌ @ int.jl:287 within `mod`
; ││┌ @ int.jl:86 within `-`
     %155 = and i32 %75, 771
; │└└
; │┌ @ promotion.jl:487 within `==` @ promotion.jl:637
    %.not41 = icmp eq i32 %155, 0
; │└
; │┌ @ essentials.jl:799 within `ifelse`
    %156 = select i1 %.not41, i32 4, i32 %155
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:86 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:44 within `DualAccessMatrix`
; │┌ @ int.jl:88 within `*`
    %157 = mul nuw nsw i32 %.zext80, 284
; │└
; │┌ @ int.jl:87 within `+`
    %158 = add nuw nsw i32 %156, %157
    %159 = shl nuw nsw i32 %77, 3
    %160 = zext nneg i32 %159 to i64
    %161 = getelementptr float, ptr addrspace(3) @shmem, i64 %160
    %162 = getelementptr float, ptr addrspace(3) %161, i64 -8
    %163 = getelementptr float, ptr addrspace(3) %161, i64 -7
    %164 = getelementptr float, ptr addrspace(3) %161, i64 -6
    %165 = getelementptr float, ptr addrspace(3) %161, i64 -5
    %166 = getelementptr float, ptr addrspace(3) %161, i64 -4
    %167 = getelementptr float, ptr addrspace(3) %161, i64 -3
    %168 = getelementptr float, ptr addrspace(3) %161, i64 -2
    %169 = getelementptr float, ptr addrspace(3) %161, i64 -1
    %170 = mul nuw nsw i32 %77, 36
    %171 = add nsw i32 %170, -41
    %172 = add nsw i32 %171, %157
    %173 = add nsw i32 %172, %156
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:89 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %174 = load float, ptr addrspace(3) %162, align 4
             %175 = load float, ptr addrspace(3) %163, align 4
             %176 = load float, ptr addrspace(3) %164, align 4
             %177 = load float, ptr addrspace(3) %165, align 4
             %178 = load float, ptr addrspace(3) %166, align 4
             %179 = load float, ptr addrspace(3) %167, align 4
             %180 = load float, ptr addrspace(3) %168, align 4
             %181 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %182 = zext nneg i32 %158 to i64
              %183 = getelementptr float, ptr addrspace(3) @shmem57, i64 %182
              %184 = getelementptr float, ptr addrspace(3) %183, i64 -1
              %185 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %186 = fmul float %174, %185
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %187 = getelementptr float, ptr addrspace(3) %183, i64 35
              %188 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %189 = fmul float %175, %188
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %190 = getelementptr float, ptr addrspace(3) %183, i64 71
              %191 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %192 = fmul float %176, %191
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %193 = getelementptr float, ptr addrspace(3) %183, i64 107
              %194 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %195 = fmul float %177, %194
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %196 = getelementptr float, ptr addrspace(3) %183, i64 143
              %197 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %198 = fmul float %178, %197
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %199 = getelementptr float, ptr addrspace(3) %183, i64 179
              %200 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %201 = fmul float %179, %200
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %202 = getelementptr float, ptr addrspace(3) %183, i64 215
              %203 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %204 = fmul float %180, %203
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %205 = getelementptr float, ptr addrspace(3) %183, i64 251
              %206 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %207 = fmul float %181, %206
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %208 = fadd float %186, %189
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %209 = fadd float %208, %192
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %210 = fadd float %209, %195
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %211 = fadd float %210, %198
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %212 = fadd float %211, %201
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %213 = fadd float %212, %204
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %214 = fadd float %213, %207
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %215 = sext i32 %173 to i64
           %216 = getelementptr float, ptr addrspace(3) @shmem60, i64 %215
           %217 = getelementptr float, ptr addrspace(3) %216, i64 4
           store float %214, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %218 = getelementptr float, ptr addrspace(3) %183, i64 3
              %219 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %220 = fmul float %174, %219
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %221 = getelementptr float, ptr addrspace(3) %183, i64 39
              %222 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %223 = fmul float %175, %222
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %224 = getelementptr float, ptr addrspace(3) %183, i64 75
              %225 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %226 = fmul float %176, %225
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %227 = getelementptr float, ptr addrspace(3) %183, i64 111
              %228 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %229 = fmul float %177, %228
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %230 = getelementptr float, ptr addrspace(3) %183, i64 147
              %231 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %232 = fmul float %178, %231
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %233 = getelementptr float, ptr addrspace(3) %183, i64 183
              %234 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %235 = fmul float %179, %234
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %236 = getelementptr float, ptr addrspace(3) %183, i64 219
              %237 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %238 = fmul float %180, %237
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %239 = getelementptr float, ptr addrspace(3) %183, i64 255
              %240 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %241 = fmul float %181, %240
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %242 = fadd float %220, %223
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %243 = fadd float %242, %226
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %244 = fadd float %243, %229
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %245 = fadd float %244, %232
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %246 = fadd float %245, %235
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %247 = fadd float %246, %238
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %248 = fadd float %247, %241
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %249 = getelementptr float, ptr addrspace(3) %216, i64 8
           store float %248, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %250 = getelementptr float, ptr addrspace(3) %183, i64 7
              %251 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %252 = fmul float %174, %251
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %253 = getelementptr float, ptr addrspace(3) %183, i64 43
              %254 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %255 = fmul float %175, %254
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %256 = getelementptr float, ptr addrspace(3) %183, i64 79
              %257 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %258 = fmul float %176, %257
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %259 = getelementptr float, ptr addrspace(3) %183, i64 115
              %260 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %261 = fmul float %177, %260
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %262 = getelementptr float, ptr addrspace(3) %183, i64 151
              %263 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %264 = fmul float %178, %263
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %265 = getelementptr float, ptr addrspace(3) %183, i64 187
              %266 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %267 = fmul float %179, %266
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %268 = getelementptr float, ptr addrspace(3) %183, i64 223
              %269 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %270 = fmul float %180, %269
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %271 = getelementptr float, ptr addrspace(3) %183, i64 259
              %272 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %273 = fmul float %181, %272
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %274 = fadd float %252, %255
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %275 = fadd float %274, %258
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %276 = fadd float %275, %261
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %277 = fadd float %276, %264
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %278 = fadd float %277, %267
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %279 = fadd float %278, %270
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %280 = fadd float %279, %273
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %281 = getelementptr float, ptr addrspace(3) %216, i64 12
           store float %280, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %282 = getelementptr float, ptr addrspace(3) %183, i64 11
              %283 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %284 = fmul float %174, %283
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %285 = getelementptr float, ptr addrspace(3) %183, i64 47
              %286 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %287 = fmul float %175, %286
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %288 = getelementptr float, ptr addrspace(3) %183, i64 83
              %289 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %290 = fmul float %176, %289
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %291 = getelementptr float, ptr addrspace(3) %183, i64 119
              %292 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %293 = fmul float %177, %292
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %294 = getelementptr float, ptr addrspace(3) %183, i64 155
              %295 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %296 = fmul float %178, %295
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %297 = getelementptr float, ptr addrspace(3) %183, i64 191
              %298 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %299 = fmul float %179, %298
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %300 = getelementptr float, ptr addrspace(3) %183, i64 227
              %301 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %302 = fmul float %180, %301
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %303 = getelementptr float, ptr addrspace(3) %183, i64 263
              %304 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %305 = fmul float %181, %304
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %306 = fadd float %284, %287
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %307 = fadd float %306, %290
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %308 = fadd float %307, %293
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %309 = fadd float %308, %296
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %310 = fadd float %309, %299
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %311 = fadd float %310, %302
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %312 = fadd float %311, %305
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %313 = getelementptr float, ptr addrspace(3) %216, i64 16
           store float %312, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %314 = load float, ptr addrspace(3) %162, align 4
             %315 = load float, ptr addrspace(3) %163, align 4
             %316 = load float, ptr addrspace(3) %164, align 4
             %317 = load float, ptr addrspace(3) %165, align 4
             %318 = load float, ptr addrspace(3) %166, align 4
             %319 = load float, ptr addrspace(3) %167, align 4
             %320 = load float, ptr addrspace(3) %168, align 4
             %321 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %322 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %323 = fmul float %314, %322
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %324 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %325 = fmul float %315, %324
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %326 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %327 = fmul float %316, %326
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %328 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %329 = fmul float %317, %328
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %330 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %331 = fmul float %318, %330
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %332 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %333 = fmul float %319, %332
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %334 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %335 = fmul float %320, %334
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %336 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %337 = fmul float %321, %336
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %338 = fadd float %323, %325
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %339 = fadd float %338, %327
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %340 = fadd float %339, %329
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %341 = fadd float %340, %331
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %342 = fadd float %341, %333
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %343 = fadd float %342, %335
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %344 = fadd float %343, %337
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %344, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %345 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %346 = fmul float %314, %345
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %347 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %348 = fmul float %315, %347
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %349 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %350 = fmul float %316, %349
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %351 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %352 = fmul float %317, %351
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %353 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %354 = fmul float %318, %353
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %355 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %356 = fmul float %319, %355
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %357 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %358 = fmul float %320, %357
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %359 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %360 = fmul float %321, %359
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %361 = fadd float %346, %348
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %362 = fadd float %361, %350
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %363 = fadd float %362, %352
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %364 = fadd float %363, %354
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %365 = fadd float %364, %356
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %366 = fadd float %365, %358
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %367 = fadd float %366, %360
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %367, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %368 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %369 = fmul float %314, %368
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %370 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %371 = fmul float %315, %370
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %372 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %373 = fmul float %316, %372
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %374 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %375 = fmul float %317, %374
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %376 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %377 = fmul float %318, %376
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %378 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %379 = fmul float %319, %378
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %380 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %381 = fmul float %320, %380
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %382 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %383 = fmul float %321, %382
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %384 = fadd float %369, %371
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %385 = fadd float %384, %373
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %386 = fadd float %385, %375
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %387 = fadd float %386, %377
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %388 = fadd float %387, %379
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %389 = fadd float %388, %381
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %390 = fadd float %389, %383
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %390, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %391 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %392 = fmul float %314, %391
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %393 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %394 = fmul float %315, %393
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %395 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %396 = fmul float %316, %395
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %397 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %398 = fmul float %317, %397
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %399 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %400 = fmul float %318, %399
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %401 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %402 = fmul float %319, %401
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %403 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %404 = fmul float %320, %403
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %405 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %406 = fmul float %321, %405
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %407 = fadd float %392, %394
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %408 = fadd float %407, %396
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %409 = fadd float %408, %398
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %410 = fadd float %409, %400
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %411 = fadd float %410, %402
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %412 = fadd float %411, %404
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %413 = fadd float %412, %406
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %413, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %414 = load float, ptr addrspace(3) %162, align 4
             %415 = load float, ptr addrspace(3) %163, align 4
             %416 = load float, ptr addrspace(3) %164, align 4
             %417 = load float, ptr addrspace(3) %165, align 4
             %418 = load float, ptr addrspace(3) %166, align 4
             %419 = load float, ptr addrspace(3) %167, align 4
             %420 = load float, ptr addrspace(3) %168, align 4
             %421 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %422 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %423 = fmul float %414, %422
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %424 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %425 = fmul float %415, %424
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %426 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %427 = fmul float %416, %426
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %428 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %429 = fmul float %417, %428
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %430 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %431 = fmul float %418, %430
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %432 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %433 = fmul float %419, %432
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %434 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %435 = fmul float %420, %434
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %436 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %437 = fmul float %421, %436
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %438 = fadd float %423, %425
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %439 = fadd float %438, %427
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %440 = fadd float %439, %429
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %441 = fadd float %440, %431
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %442 = fadd float %441, %433
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %443 = fadd float %442, %435
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %444 = fadd float %443, %437
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %444, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %445 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %446 = fmul float %414, %445
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %447 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %448 = fmul float %415, %447
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %449 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %450 = fmul float %416, %449
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %451 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %452 = fmul float %417, %451
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %453 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %454 = fmul float %418, %453
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %455 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %456 = fmul float %419, %455
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %457 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %458 = fmul float %420, %457
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %459 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %460 = fmul float %421, %459
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %461 = fadd float %446, %448
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %462 = fadd float %461, %450
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %463 = fadd float %462, %452
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %464 = fadd float %463, %454
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %465 = fadd float %464, %456
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %466 = fadd float %465, %458
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %467 = fadd float %466, %460
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %467, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %468 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %469 = fmul float %414, %468
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %470 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %471 = fmul float %415, %470
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %472 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %473 = fmul float %416, %472
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %474 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %475 = fmul float %417, %474
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %476 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %477 = fmul float %418, %476
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %478 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %479 = fmul float %419, %478
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %480 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %481 = fmul float %420, %480
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %482 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %483 = fmul float %421, %482
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %484 = fadd float %469, %471
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %485 = fadd float %484, %473
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %486 = fadd float %485, %475
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %487 = fadd float %486, %477
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %488 = fadd float %487, %479
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %489 = fadd float %488, %481
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %490 = fadd float %489, %483
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %490, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %491 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %492 = fmul float %414, %491
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %493 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %494 = fmul float %415, %493
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %495 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %496 = fmul float %416, %495
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %497 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %498 = fmul float %417, %497
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %499 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %500 = fmul float %418, %499
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %501 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %502 = fmul float %419, %501
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %503 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %504 = fmul float %420, %503
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %505 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %506 = fmul float %421, %505
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %507 = fadd float %492, %494
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %508 = fadd float %507, %496
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %509 = fadd float %508, %498
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %510 = fadd float %509, %500
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %511 = fadd float %510, %502
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %512 = fadd float %511, %504
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %513 = fadd float %512, %506
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %513, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %514 = load float, ptr addrspace(3) %162, align 4
             %515 = load float, ptr addrspace(3) %163, align 4
             %516 = load float, ptr addrspace(3) %164, align 4
             %517 = load float, ptr addrspace(3) %165, align 4
             %518 = load float, ptr addrspace(3) %166, align 4
             %519 = load float, ptr addrspace(3) %167, align 4
             %520 = load float, ptr addrspace(3) %168, align 4
             %521 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %522 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %523 = fmul float %514, %522
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %524 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %525 = fmul float %515, %524
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %526 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %527 = fmul float %516, %526
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %528 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %529 = fmul float %517, %528
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %530 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %531 = fmul float %518, %530
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %532 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %533 = fmul float %519, %532
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %534 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %535 = fmul float %520, %534
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %536 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %537 = fmul float %521, %536
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %538 = fadd float %523, %525
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %539 = fadd float %538, %527
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %540 = fadd float %539, %529
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %541 = fadd float %540, %531
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %542 = fadd float %541, %533
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %543 = fadd float %542, %535
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %544 = fadd float %543, %537
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %544, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %545 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %546 = fmul float %514, %545
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %547 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %548 = fmul float %515, %547
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %549 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %550 = fmul float %516, %549
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %551 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %552 = fmul float %517, %551
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %553 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %554 = fmul float %518, %553
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %555 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %556 = fmul float %519, %555
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %557 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %558 = fmul float %520, %557
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %559 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %560 = fmul float %521, %559
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %561 = fadd float %546, %548
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %562 = fadd float %561, %550
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %563 = fadd float %562, %552
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %564 = fadd float %563, %554
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %565 = fadd float %564, %556
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %566 = fadd float %565, %558
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %567 = fadd float %566, %560
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %567, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %568 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %569 = fmul float %514, %568
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %570 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %571 = fmul float %515, %570
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %572 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %573 = fmul float %516, %572
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %574 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %575 = fmul float %517, %574
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %576 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %577 = fmul float %518, %576
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %578 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %579 = fmul float %519, %578
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %580 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %581 = fmul float %520, %580
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %582 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %583 = fmul float %521, %582
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %584 = fadd float %569, %571
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %585 = fadd float %584, %573
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %586 = fadd float %585, %575
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %587 = fadd float %586, %577
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %588 = fadd float %587, %579
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %589 = fadd float %588, %581
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %590 = fadd float %589, %583
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %590, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %591 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %592 = fmul float %514, %591
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %593 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %594 = fmul float %515, %593
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %595 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %596 = fmul float %516, %595
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %597 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %598 = fmul float %517, %597
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %599 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %600 = fmul float %518, %599
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %601 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %602 = fmul float %519, %601
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %603 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %604 = fmul float %520, %603
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %605 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %606 = fmul float %521, %605
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %607 = fadd float %592, %594
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %608 = fadd float %607, %596
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %609 = fadd float %608, %598
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %610 = fadd float %609, %600
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %611 = fadd float %610, %602
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %612 = fadd float %611, %604
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %613 = fadd float %612, %606
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %613, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %614 = load float, ptr addrspace(3) %162, align 4
             %615 = load float, ptr addrspace(3) %163, align 4
             %616 = load float, ptr addrspace(3) %164, align 4
             %617 = load float, ptr addrspace(3) %165, align 4
             %618 = load float, ptr addrspace(3) %166, align 4
             %619 = load float, ptr addrspace(3) %167, align 4
             %620 = load float, ptr addrspace(3) %168, align 4
             %621 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %622 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %623 = fmul float %614, %622
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %624 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %625 = fmul float %615, %624
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %626 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %627 = fmul float %616, %626
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %628 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %629 = fmul float %617, %628
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %630 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %631 = fmul float %618, %630
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %632 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %633 = fmul float %619, %632
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %634 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %635 = fmul float %620, %634
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %636 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %637 = fmul float %621, %636
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %638 = fadd float %623, %625
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %639 = fadd float %638, %627
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %640 = fadd float %639, %629
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %641 = fadd float %640, %631
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %642 = fadd float %641, %633
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %643 = fadd float %642, %635
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %644 = fadd float %643, %637
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %644, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %645 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %646 = fmul float %614, %645
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %647 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %648 = fmul float %615, %647
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %649 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %650 = fmul float %616, %649
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %651 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %652 = fmul float %617, %651
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %653 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %654 = fmul float %618, %653
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %655 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %656 = fmul float %619, %655
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %657 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %658 = fmul float %620, %657
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %659 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %660 = fmul float %621, %659
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %661 = fadd float %646, %648
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %662 = fadd float %661, %650
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %663 = fadd float %662, %652
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %664 = fadd float %663, %654
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %665 = fadd float %664, %656
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %666 = fadd float %665, %658
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %667 = fadd float %666, %660
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %667, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %668 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %669 = fmul float %614, %668
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %670 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %671 = fmul float %615, %670
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %672 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %673 = fmul float %616, %672
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %674 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %675 = fmul float %617, %674
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %676 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %677 = fmul float %618, %676
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %678 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %679 = fmul float %619, %678
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %680 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %681 = fmul float %620, %680
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %682 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %683 = fmul float %621, %682
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %684 = fadd float %669, %671
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %685 = fadd float %684, %673
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %686 = fadd float %685, %675
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %687 = fadd float %686, %677
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %688 = fadd float %687, %679
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %689 = fadd float %688, %681
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %690 = fadd float %689, %683
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %690, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %691 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %692 = fmul float %614, %691
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %693 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %694 = fmul float %615, %693
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %695 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %696 = fmul float %616, %695
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %697 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %698 = fmul float %617, %697
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %699 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %700 = fmul float %618, %699
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %701 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %702 = fmul float %619, %701
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %703 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %704 = fmul float %620, %703
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %705 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %706 = fmul float %621, %705
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %707 = fadd float %692, %694
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %708 = fadd float %707, %696
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %709 = fadd float %708, %698
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %710 = fadd float %709, %700
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %711 = fadd float %710, %702
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %712 = fadd float %711, %704
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %713 = fadd float %712, %706
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %713, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %714 = load float, ptr addrspace(3) %162, align 4
             %715 = load float, ptr addrspace(3) %163, align 4
             %716 = load float, ptr addrspace(3) %164, align 4
             %717 = load float, ptr addrspace(3) %165, align 4
             %718 = load float, ptr addrspace(3) %166, align 4
             %719 = load float, ptr addrspace(3) %167, align 4
             %720 = load float, ptr addrspace(3) %168, align 4
             %721 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %722 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %723 = fmul float %714, %722
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %724 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %725 = fmul float %715, %724
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %726 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %727 = fmul float %716, %726
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %728 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %729 = fmul float %717, %728
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %730 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %731 = fmul float %718, %730
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %732 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %733 = fmul float %719, %732
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %734 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %735 = fmul float %720, %734
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %736 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %737 = fmul float %721, %736
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %738 = fadd float %723, %725
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %739 = fadd float %738, %727
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %740 = fadd float %739, %729
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %741 = fadd float %740, %731
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %742 = fadd float %741, %733
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %743 = fadd float %742, %735
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %744 = fadd float %743, %737
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %744, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %745 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %746 = fmul float %714, %745
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %747 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %748 = fmul float %715, %747
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %749 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %750 = fmul float %716, %749
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %751 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %752 = fmul float %717, %751
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %753 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %754 = fmul float %718, %753
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %755 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %756 = fmul float %719, %755
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %757 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %758 = fmul float %720, %757
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %759 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %760 = fmul float %721, %759
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %761 = fadd float %746, %748
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %762 = fadd float %761, %750
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %763 = fadd float %762, %752
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %764 = fadd float %763, %754
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %765 = fadd float %764, %756
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %766 = fadd float %765, %758
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %767 = fadd float %766, %760
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %767, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %768 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %769 = fmul float %714, %768
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %770 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %771 = fmul float %715, %770
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %772 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %773 = fmul float %716, %772
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %774 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %775 = fmul float %717, %774
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %776 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %777 = fmul float %718, %776
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %778 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %779 = fmul float %719, %778
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %780 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %781 = fmul float %720, %780
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %782 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %783 = fmul float %721, %782
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %784 = fadd float %769, %771
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %785 = fadd float %784, %773
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %786 = fadd float %785, %775
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %787 = fadd float %786, %777
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %788 = fadd float %787, %779
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %789 = fadd float %788, %781
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %790 = fadd float %789, %783
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %790, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %791 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %792 = fmul float %714, %791
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %793 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %794 = fmul float %715, %793
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %795 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %796 = fmul float %716, %795
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %797 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %798 = fmul float %717, %797
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %799 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %800 = fmul float %718, %799
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %801 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %802 = fmul float %719, %801
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %803 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %804 = fmul float %720, %803
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %805 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %806 = fmul float %721, %805
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %807 = fadd float %792, %794
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %808 = fadd float %807, %796
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %809 = fadd float %808, %798
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %810 = fadd float %809, %800
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %811 = fadd float %810, %802
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %812 = fadd float %811, %804
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %813 = fadd float %812, %806
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %813, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %814 = load float, ptr addrspace(3) %162, align 4
             %815 = load float, ptr addrspace(3) %163, align 4
             %816 = load float, ptr addrspace(3) %164, align 4
             %817 = load float, ptr addrspace(3) %165, align 4
             %818 = load float, ptr addrspace(3) %166, align 4
             %819 = load float, ptr addrspace(3) %167, align 4
             %820 = load float, ptr addrspace(3) %168, align 4
             %821 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %822 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %823 = fmul float %814, %822
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %824 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %825 = fmul float %815, %824
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %826 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %827 = fmul float %816, %826
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %828 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %829 = fmul float %817, %828
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %830 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %831 = fmul float %818, %830
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %832 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %833 = fmul float %819, %832
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %834 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %835 = fmul float %820, %834
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %836 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %837 = fmul float %821, %836
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %838 = fadd float %823, %825
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %839 = fadd float %838, %827
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %840 = fadd float %839, %829
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %841 = fadd float %840, %831
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %842 = fadd float %841, %833
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %843 = fadd float %842, %835
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %844 = fadd float %843, %837
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %844, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %845 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %846 = fmul float %814, %845
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %847 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %848 = fmul float %815, %847
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %849 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %850 = fmul float %816, %849
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %851 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %852 = fmul float %817, %851
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %853 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %854 = fmul float %818, %853
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %855 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %856 = fmul float %819, %855
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %857 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %858 = fmul float %820, %857
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %859 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %860 = fmul float %821, %859
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %861 = fadd float %846, %848
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %862 = fadd float %861, %850
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %863 = fadd float %862, %852
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %864 = fadd float %863, %854
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %865 = fadd float %864, %856
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %866 = fadd float %865, %858
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %867 = fadd float %866, %860
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %867, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %868 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %869 = fmul float %814, %868
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %870 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %871 = fmul float %815, %870
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %872 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %873 = fmul float %816, %872
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %874 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %875 = fmul float %817, %874
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %876 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %877 = fmul float %818, %876
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %878 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %879 = fmul float %819, %878
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %880 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %881 = fmul float %820, %880
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %882 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %883 = fmul float %821, %882
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %884 = fadd float %869, %871
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %885 = fadd float %884, %873
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %886 = fadd float %885, %875
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %887 = fadd float %886, %877
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %888 = fadd float %887, %879
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %889 = fadd float %888, %881
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %890 = fadd float %889, %883
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %890, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %891 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %892 = fmul float %814, %891
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %893 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %894 = fmul float %815, %893
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %895 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %896 = fmul float %816, %895
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %897 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %898 = fmul float %817, %897
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %899 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %900 = fmul float %818, %899
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %901 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %902 = fmul float %819, %901
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %903 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %904 = fmul float %820, %903
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %905 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %906 = fmul float %821, %905
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %907 = fadd float %892, %894
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %908 = fadd float %907, %896
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %909 = fadd float %908, %898
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %910 = fadd float %909, %900
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %911 = fadd float %910, %902
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %912 = fadd float %911, %904
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %913 = fadd float %912, %906
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %913, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %914 = load float, ptr addrspace(3) %162, align 4
             %915 = load float, ptr addrspace(3) %163, align 4
             %916 = load float, ptr addrspace(3) %164, align 4
             %917 = load float, ptr addrspace(3) %165, align 4
             %918 = load float, ptr addrspace(3) %166, align 4
             %919 = load float, ptr addrspace(3) %167, align 4
             %920 = load float, ptr addrspace(3) %168, align 4
             %921 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %922 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %923 = fmul float %914, %922
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %924 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %925 = fmul float %915, %924
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %926 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %927 = fmul float %916, %926
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %928 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %929 = fmul float %917, %928
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %930 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %931 = fmul float %918, %930
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %932 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %933 = fmul float %919, %932
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %934 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %935 = fmul float %920, %934
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %936 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %937 = fmul float %921, %936
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %938 = fadd float %923, %925
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %939 = fadd float %938, %927
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %940 = fadd float %939, %929
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %941 = fadd float %940, %931
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %942 = fadd float %941, %933
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %943 = fadd float %942, %935
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %944 = fadd float %943, %937
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %944, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %945 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %946 = fmul float %914, %945
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %947 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %948 = fmul float %915, %947
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %949 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %950 = fmul float %916, %949
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %951 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %952 = fmul float %917, %951
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %953 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %954 = fmul float %918, %953
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %955 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %956 = fmul float %919, %955
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %957 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %958 = fmul float %920, %957
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %959 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %960 = fmul float %921, %959
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %961 = fadd float %946, %948
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %962 = fadd float %961, %950
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %963 = fadd float %962, %952
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %964 = fadd float %963, %954
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %965 = fadd float %964, %956
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %966 = fadd float %965, %958
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %967 = fadd float %966, %960
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %967, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %968 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %969 = fmul float %914, %968
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %970 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %971 = fmul float %915, %970
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %972 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %973 = fmul float %916, %972
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %974 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %975 = fmul float %917, %974
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %976 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %977 = fmul float %918, %976
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %978 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %979 = fmul float %919, %978
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %980 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %981 = fmul float %920, %980
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %982 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %983 = fmul float %921, %982
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %984 = fadd float %969, %971
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %985 = fadd float %984, %973
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %986 = fadd float %985, %975
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %987 = fadd float %986, %977
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %988 = fadd float %987, %979
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %989 = fadd float %988, %981
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %990 = fadd float %989, %983
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %990, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %991 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %992 = fmul float %914, %991
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %993 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %994 = fmul float %915, %993
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %995 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %996 = fmul float %916, %995
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %997 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %998 = fmul float %917, %997
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %999 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1000 = fmul float %918, %999
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1001 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1002 = fmul float %919, %1001
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1003 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1004 = fmul float %920, %1003
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1005 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1006 = fmul float %921, %1005
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1007 = fadd float %992, %994
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1008 = fadd float %1007, %996
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1009 = fadd float %1008, %998
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1010 = fadd float %1009, %1000
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1011 = fadd float %1010, %1002
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1012 = fadd float %1011, %1004
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1013 = fadd float %1012, %1006
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1013, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %1014 = load float, ptr addrspace(3) %162, align 4
             %1015 = load float, ptr addrspace(3) %163, align 4
             %1016 = load float, ptr addrspace(3) %164, align 4
             %1017 = load float, ptr addrspace(3) %165, align 4
             %1018 = load float, ptr addrspace(3) %166, align 4
             %1019 = load float, ptr addrspace(3) %167, align 4
             %1020 = load float, ptr addrspace(3) %168, align 4
             %1021 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1022 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1023 = fmul float %1014, %1022
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1024 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1025 = fmul float %1015, %1024
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1026 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1027 = fmul float %1016, %1026
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1028 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1029 = fmul float %1017, %1028
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1030 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1031 = fmul float %1018, %1030
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1032 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1033 = fmul float %1019, %1032
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1034 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1035 = fmul float %1020, %1034
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1036 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1037 = fmul float %1021, %1036
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1038 = fadd float %1023, %1025
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1039 = fadd float %1038, %1027
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1040 = fadd float %1039, %1029
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1041 = fadd float %1040, %1031
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1042 = fadd float %1041, %1033
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1043 = fadd float %1042, %1035
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1044 = fadd float %1043, %1037
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1044, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1045 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1046 = fmul float %1014, %1045
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1047 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1048 = fmul float %1015, %1047
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1049 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1050 = fmul float %1016, %1049
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1051 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1052 = fmul float %1017, %1051
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1053 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1054 = fmul float %1018, %1053
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1055 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1056 = fmul float %1019, %1055
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1057 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1058 = fmul float %1020, %1057
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1059 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1060 = fmul float %1021, %1059
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1061 = fadd float %1046, %1048
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1062 = fadd float %1061, %1050
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1063 = fadd float %1062, %1052
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1064 = fadd float %1063, %1054
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1065 = fadd float %1064, %1056
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1066 = fadd float %1065, %1058
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1067 = fadd float %1066, %1060
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1067, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1068 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1069 = fmul float %1014, %1068
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1070 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1071 = fmul float %1015, %1070
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1072 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1073 = fmul float %1016, %1072
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1074 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1075 = fmul float %1017, %1074
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1076 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1077 = fmul float %1018, %1076
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1078 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1079 = fmul float %1019, %1078
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1080 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1081 = fmul float %1020, %1080
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1082 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1083 = fmul float %1021, %1082
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1084 = fadd float %1069, %1071
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1085 = fadd float %1084, %1073
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1086 = fadd float %1085, %1075
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1087 = fadd float %1086, %1077
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1088 = fadd float %1087, %1079
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1089 = fadd float %1088, %1081
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1090 = fadd float %1089, %1083
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1090, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1091 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1092 = fmul float %1014, %1091
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1093 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1094 = fmul float %1015, %1093
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1095 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1096 = fmul float %1016, %1095
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1097 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1098 = fmul float %1017, %1097
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1099 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1100 = fmul float %1018, %1099
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1101 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1102 = fmul float %1019, %1101
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1103 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1104 = fmul float %1020, %1103
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1105 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1106 = fmul float %1021, %1105
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1107 = fadd float %1092, %1094
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1108 = fadd float %1107, %1096
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1109 = fadd float %1108, %1098
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1110 = fadd float %1109, %1100
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1111 = fadd float %1110, %1102
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1112 = fadd float %1111, %1104
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1113 = fadd float %1112, %1106
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1113, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %1114 = load float, ptr addrspace(3) %162, align 4
             %1115 = load float, ptr addrspace(3) %163, align 4
             %1116 = load float, ptr addrspace(3) %164, align 4
             %1117 = load float, ptr addrspace(3) %165, align 4
             %1118 = load float, ptr addrspace(3) %166, align 4
             %1119 = load float, ptr addrspace(3) %167, align 4
             %1120 = load float, ptr addrspace(3) %168, align 4
             %1121 = load float, ptr addrspace(3) %169, align 4
; ││└└└└└└└└└
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:163 within `batch_op!`
; ││┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1122 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1123 = fmul float %1114, %1122
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1124 = load float, ptr addrspace(3) %187, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1125 = fmul float %1115, %1124
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1126 = load float, ptr addrspace(3) %190, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1127 = fmul float %1116, %1126
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1128 = load float, ptr addrspace(3) %193, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1129 = fmul float %1117, %1128
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1130 = load float, ptr addrspace(3) %196, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1131 = fmul float %1118, %1130
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1132 = load float, ptr addrspace(3) %199, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1133 = fmul float %1119, %1132
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1134 = load float, ptr addrspace(3) %202, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1135 = fmul float %1120, %1134
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1136 = load float, ptr addrspace(3) %205, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1137 = fmul float %1121, %1136
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1138 = fadd float %1123, %1125
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1139 = fadd float %1138, %1127
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1140 = fadd float %1139, %1129
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1141 = fadd float %1140, %1131
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1142 = fadd float %1141, %1133
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1143 = fadd float %1142, %1135
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1144 = fadd float %1143, %1137
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1144, ptr addrspace(3) %217, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1145 = load float, ptr addrspace(3) %218, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1146 = fmul float %1114, %1145
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1147 = load float, ptr addrspace(3) %221, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1148 = fmul float %1115, %1147
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1149 = load float, ptr addrspace(3) %224, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1150 = fmul float %1116, %1149
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1151 = load float, ptr addrspace(3) %227, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1152 = fmul float %1117, %1151
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1153 = load float, ptr addrspace(3) %230, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1154 = fmul float %1118, %1153
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1155 = load float, ptr addrspace(3) %233, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1156 = fmul float %1119, %1155
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1157 = load float, ptr addrspace(3) %236, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1158 = fmul float %1120, %1157
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1159 = load float, ptr addrspace(3) %239, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1160 = fmul float %1121, %1159
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1161 = fadd float %1146, %1148
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1162 = fadd float %1161, %1150
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1163 = fadd float %1162, %1152
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1164 = fadd float %1163, %1154
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1165 = fadd float %1164, %1156
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1166 = fadd float %1165, %1158
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1167 = fadd float %1166, %1160
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1167, ptr addrspace(3) %249, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1168 = load float, ptr addrspace(3) %250, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1169 = fmul float %1114, %1168
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1170 = load float, ptr addrspace(3) %253, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1171 = fmul float %1115, %1170
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1172 = load float, ptr addrspace(3) %256, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1173 = fmul float %1116, %1172
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1174 = load float, ptr addrspace(3) %259, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1175 = fmul float %1117, %1174
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1176 = load float, ptr addrspace(3) %262, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1177 = fmul float %1118, %1176
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1178 = load float, ptr addrspace(3) %265, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1179 = fmul float %1119, %1178
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1180 = load float, ptr addrspace(3) %268, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1181 = fmul float %1120, %1180
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1182 = load float, ptr addrspace(3) %271, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1183 = fmul float %1121, %1182
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1184 = fadd float %1169, %1171
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1185 = fadd float %1184, %1173
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1186 = fadd float %1185, %1175
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1187 = fadd float %1186, %1177
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1188 = fadd float %1187, %1179
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1189 = fadd float %1188, %1181
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1190 = fadd float %1189, %1183
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1190, ptr addrspace(3) %281, align 4
; │││└└└└└└
; │││┌ @ ntuple.jl:71 within `ntuple`
; ││││┌ @ ntuple.jl:74 within `macro expansion`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:164 within `#batch_op!##2`
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1191 = load float, ptr addrspace(3) %282, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1192 = fmul float %1114, %1191
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1193 = load float, ptr addrspace(3) %285, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1194 = fmul float %1115, %1193
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1195 = load float, ptr addrspace(3) %288, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1196 = fmul float %1116, %1195
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1197 = load float, ptr addrspace(3) %291, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1198 = fmul float %1117, %1197
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1199 = load float, ptr addrspace(3) %294, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1200 = fmul float %1118, %1199
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1201 = load float, ptr addrspace(3) %297, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1202 = fmul float %1119, %1201
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1203 = load float, ptr addrspace(3) %300, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1204 = fmul float %1120, %1203
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1205 = load float, ptr addrspace(3) %303, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1206 = fmul float %1121, %1205
; │││└└└└
; │││┌ @ reduce.jl:553 within `sum`
; ││││┌ @ reduce.jl:553 within `#sum#278`
; │││││┌ @ reduce.jl:524 within `sum`
; ││││││┌ @ reduce.jl:524 within `#sum#277`
; │││││││┌ @ reduce.jl:299 within `mapreduce`
; ││││││││┌ @ reduce.jl:299 within `#mapreduce#274`
; │││││││││┌ @ reduce.jl:167 within `mapfoldl`
; ││││││││││┌ @ reduce.jl:167 within `#mapfoldl#270`
; │││││││││││┌ @ reduce.jl:36 within `mapfoldl_impl`
; ││││││││││││┌ @ reduce.jl:40 within `foldl_impl`
; │││││││││││││┌ @ reduce.jl:60 within `_foldl_impl`
; ││││││││││││││┌ @ operators.jl:600 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1207 = fadd float %1192, %1194
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1208 = fadd float %1207, %1196
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1209 = fadd float %1208, %1198
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1210 = fadd float %1209, %1200
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1211 = fadd float %1210, %1202
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1212 = fadd float %1211, %1204
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1213 = fadd float %1212, %1206
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1213, ptr addrspace(3) %313, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:95 within `kernel_mul_defrag!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   br label %L866

L946.preheader:                                   ; preds = %L866
   %1214 = shl nuw nsw i32 %.zext55, 4
   %1215 = shl nuw nsw i32 %104, 2
   %1216 = add nsw i32 %1215, -4
   %1217 = add nsw i32 %1216, %1214
   %1218 = mul nuw nsw i32 %104, 36
   %1219 = add nsw i32 %83, -41
   %1220 = add nsw i32 %1219, %67
   %1221 = add nsw i32 %1220, %1218
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:98 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1364 within `dual_to_interm_transfer!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc72 = trunc i32 %1217 to i8
     %1222 = sdiv i8 %.lhs.trunc72, 32
     %.sext = sext i8 %1222 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1223 = sext i32 %1221 to i64
          %1224 = getelementptr float, ptr addrspace(3) @shmem60, i64 %1223
          %1225 = getelementptr float, ptr addrspace(3) %1224, i64 4
          %1226 = load float, ptr addrspace(3) %1225, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1227 = add nsw i32 %1217, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1228 = add nsw i32 %1227, %.sext
; ││││││││└
          %1229 = sext i32 %1228 to i64
          %1230 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %1229
          store float %1226, ptr addrspace(3) %1230, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1231 = or disjoint i32 %1217, 1
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc73 = trunc i32 %1231 to i8
     %1232 = sdiv i8 %.lhs.trunc73, 32
     %.sext74 = sext i8 %1232 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1233 = getelementptr float, ptr addrspace(3) %1224, i64 8
          %1234 = load float, ptr addrspace(3) %1233, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1235 = add nsw i32 %1231, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1236 = add nsw i32 %1235, %.sext74
; ││││││││└
          %1237 = sext i32 %1236 to i64
          %1238 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %1237
          store float %1234, ptr addrspace(3) %1238, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1239 = or disjoint i32 %1217, 2
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc75 = trunc i32 %1239 to i8
     %1240 = sdiv i8 %.lhs.trunc75, 32
     %.sext76 = sext i8 %1240 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1241 = getelementptr float, ptr addrspace(3) %1224, i64 12
          %1242 = load float, ptr addrspace(3) %1241, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1243 = add nsw i32 %1239, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1244 = add nsw i32 %1243, %.sext76
; ││││││││└
          %1245 = sext i32 %1244 to i64
          %1246 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %1245
          store float %1242, ptr addrspace(3) %1246, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1247 = or disjoint i32 %1217, 3
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc77 = trunc i32 %1247 to i8
     %1248 = sdiv i8 %.lhs.trunc77, 32
     %.sext78 = sext i8 %1248 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1249 = getelementptr float, ptr addrspace(3) %1224, i64 16
          %1250 = load float, ptr addrspace(3) %1249, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1251 = add nsw i32 %1247, %83
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1252 = add nsw i32 %1251, %.sext78
; ││││││││└
          %1253 = sext i32 %1252 to i64
          %1254 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %1253
          store float %1250, ptr addrspace(3) %1254, align 4
; └└└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_defrag.jl:99 within `kernel_mul_defrag!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1417 within `intermediate_layout_write!`
; │┌ @ int.jl:301 within `div`
    br label %pass248

pass248:                                          ; preds = %L946.preheader, %L866
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1424 within `intermediate_layout_write!`
; │┌ @ int.jl:88 within `*`
    %1255 = shl nuw nsw i32 %80, 6
; │└
; │┌ @ int.jl:87 within `+`
    %1256 = or disjoint i32 %1255, 1
    %1257 = add nuw nsw i32 %1256, %69
    %1258 = add nuw nsw i32 %1255, 64
    %1259 = shl i32 %68, 9
    %1260 = add i32 %1259, -1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1429 within `intermediate_layout_write!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %1261 = add nsw i32 %1257, -1
     %1262 = add nsw i32 %1257, -2
; ││└
; ││┌ @ int.jl:301 within `div`
     %1263 = lshr i32 %1262, 4
     %.zext69 = and i32 %1263, 4095
; ││└
; ││┌ @ int.jl:87 within `+`
     %1264 = add i32 %85, %.zext69
; ││└
; ││┌ @ int.jl:520 within `<=`
     %1265 = icmp ugt i32 %1262, 511
     %.not49 = icmp sgt i32 %1264, %"N::Int32"
; ││└
    %or.cond17 = select i1 %1265, i1 true, i1 %.not49
    %.not50 = icmp ugt i32 %1261, %1258
    %or.cond99 = select i1 %or.cond17, i1 true, i1 %.not50
    br i1 %or.cond99, label %L1109, label %pass259

pass259:                                          ; preds = %pass248
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1445
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1266 = add i32 %1260, %1261
; ││││││││└
          %1267 = sext i32 %1266 to i64
          %1268 = getelementptr inbounds float, ptr addrspace(1) %"M_out::CuDeviceArray.fca.0.extract", i64 %1267
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1269 = zext nneg i32 %89 to i64
          %1270 = getelementptr inbounds float, ptr addrspace(3) @shmem57, i64 %1269
          %1271 = load float, ptr addrspace(3) %1270, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %1271, ptr addrspace(1) %1268, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L1109
; └└└└
}
