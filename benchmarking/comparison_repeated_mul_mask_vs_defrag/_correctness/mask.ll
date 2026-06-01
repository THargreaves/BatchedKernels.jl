; PTX CompilerJob of MethodInstance for kernel_mul_mask!(::CuDeviceArray{Float32, 3, 1}, ::CuDeviceArray{Float32, 3, 1}, ::CuDeviceMatrix{Float32, 1}, ::Val{4}, ::Val{8}, ::Val{8}, ::Val{256}, ::Val{10}, ::Int32) for sm_89
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:44 within `kernel_mul_mask!`
define ptx_kernel void @_Z16kernel_mul_mask_13CuDeviceArrayI7Float32Li3ELi1EES1_S_IS0_Li2ELi1EE3ValILi4EES3_ILi8EES5_S3_ILi256EES3_ILi10EE5Int32({ ptr, i32 } %state, { ptr addrspace(1), i64, [3 x i64], i64 } %"M_out::CuDeviceArray", { ptr addrspace(1), i64, [3 x i64], i64 } %"M_in::CuDeviceArray", { ptr addrspace(1), i64, [2 x i64], i64 } %"A_global::CuDeviceArray", i32 signext %"N::Int32") local_unnamed_addr {
conversion:
  %"M_out::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [3 x i64], i64 } %"M_out::CuDeviceArray", 0
  %"M_in::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [3 x i64], i64 } %"M_in::CuDeviceArray", 0
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:55 within `kernel_mul_mask!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `_index`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
      %0 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x()
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:63 within `kernel_mul_mask!`
; ┌ @ promotion.jl:637 within `==`
   %1 = icmp ugt i32 %0, 31
; └
  br i1 %1, label %conversion.pass30_crit_edge, label %pass13

conversion.pass30_crit_edge:                      ; preds = %conversion
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:69 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:26 within `get_shmem_elems`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; │││┌ @ int.jl:87 within `+`
      %.pre = add nuw nsw i32 %0, 1
; │└└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %.pre1 = and i32 %.pre, 31
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:63 within `kernel_mul_mask!`
  br label %pass30

L277:                                             ; preds = %pass107, %pass30
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:78 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1191 within `intermediate_layout_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %2 = add nuw nsw i32 %1052, 31
     %3 = add nuw nsw i32 %1052, 30
     %4 = lshr i32 %3, 5
     %.zext54 = and i32 %4, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %5 = add i32 %1044, %.zext54
; ││└
; ││┌ @ int.jl:520 within `<=`
     %6 = icmp ugt i32 %1052, 993
     %.not28.1 = icmp sgt i32 %5, %"N::Int32"
; ││└
    %or.cond.1 = select i1 %6, i1 true, i1 %.not28.1
    %.not29.1 = icmp ugt i32 %2, %1053
    %or.cond79 = select i1 %or.cond.1, i1 true, i1 %.not29.1
    br i1 %or.cond79, label %L277.1, label %pass107.1

pass107.1:                                        ; preds = %L277
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %7 = sext i32 %1056 to i64
          %8 = getelementptr float, ptr addrspace(3) @shmem42, i64 %7
          %9 = getelementptr float, ptr addrspace(3) %8, i64 33
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %10 = add i32 %1055, %2
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
      br label %L277.1

L277.1:                                           ; preds = %pass107.1, %L277
; ││└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %14 = add nuw nsw i32 %1052, 63
     %15 = add nuw nsw i32 %1052, 62
     %16 = lshr i32 %15, 5
     %.zext56 = and i32 %16, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %17 = add i32 %1044, %.zext56
; ││└
; ││┌ @ int.jl:520 within `<=`
     %18 = icmp ugt i32 %1052, 961
     %.not28.2 = icmp sgt i32 %17, %"N::Int32"
; ││└
    %or.cond.2 = select i1 %18, i1 true, i1 %.not28.2
    %.not29.2 = icmp ugt i32 %14, %1053
    %or.cond80 = select i1 %or.cond.2, i1 true, i1 %.not29.2
    br i1 %or.cond80, label %L277.2, label %pass107.2

pass107.2:                                        ; preds = %L277.1
    %.lhs.trunc76 = add nuw nsw i32 %1036, 63
    %.zext77 = lshr i32 %.lhs.trunc76, 5
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %19 = add i32 %1055, %14
; ││││││││└
          %20 = sext i32 %19 to i64
          %21 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %20
          %22 = load float, ptr addrspace(1) %21, align 4
; ││└└└└└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1205
; ││┌ @ int.jl:87 within `+`
     %23 = add nuw nsw i32 %1056, 64
; ││└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %24 = add nuw nsw i32 %23, %.zext77
           %25 = zext nneg i32 %24 to i64
; ││││││││└
          %26 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %25
          store float %22, ptr addrspace(3) %26, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L277.2

L277.2:                                           ; preds = %pass107.2, %L277.1
; ││└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %27 = add nuw nsw i32 %1052, 95
     %28 = add nuw nsw i32 %1052, 94
     %29 = lshr i32 %28, 5
     %.zext58 = and i32 %29, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %30 = add i32 %1044, %.zext58
; ││└
; ││┌ @ int.jl:520 within `<=`
     %31 = icmp ugt i32 %1052, 929
     %.not28.3 = icmp sgt i32 %30, %"N::Int32"
; ││└
    %or.cond.3 = select i1 %31, i1 true, i1 %.not28.3
    %.not29.3 = icmp ugt i32 %27, %1053
    %or.cond81 = select i1 %or.cond.3, i1 true, i1 %.not29.3
    br i1 %or.cond81, label %pass143, label %pass107.3

pass107.3:                                        ; preds = %L277.2
    %.lhs.trunc74 = add nuw nsw i32 %1036, 95
    %.zext75 = lshr i32 %.lhs.trunc74, 5
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %32 = add i32 %1055, %27
; ││││││││└
          %33 = sext i32 %32 to i64
          %34 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %33
          %35 = load float, ptr addrspace(1) %34, align 4
; ││└└└└└└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1205
; ││┌ @ int.jl:87 within `+`
     %36 = add nuw nsw i32 %1056, 96
; ││└
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %37 = add nuw nsw i32 %36, %.zext75
           %38 = zext nneg i32 %37 to i64
; ││││││││└
          %39 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %38
          store float %35, ptr addrspace(3) %39, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %pass143

L422:                                             ; preds = %L355.preheader, %pass143
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:81 within `kernel_mul_mask!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   call void @llvm.nvvm.barrier0()
   %.not35 = icmp sgt i32 %1046, %"N::Int32"
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:83 within `kernel_mul_mask!`
  br i1 %.not35, label %L826, label %L428.preheader

L428.preheader:                                   ; preds = %L422
  %40 = icmp ult i32 %1042, 5
  %41 = shl nuw nsw i32 %1042, 3
  %42 = zext nneg i32 %41 to i64
  %43 = getelementptr float, ptr addrspace(3) @shmem, i64 %42
  %44 = getelementptr float, ptr addrspace(3) %43, i64 -8
  %45 = getelementptr float, ptr addrspace(3) %43, i64 -7
  %46 = getelementptr float, ptr addrspace(3) %43, i64 -6
  %47 = getelementptr float, ptr addrspace(3) %43, i64 -5
  %48 = getelementptr float, ptr addrspace(3) %43, i64 -4
  %49 = getelementptr float, ptr addrspace(3) %43, i64 -3
  %50 = getelementptr float, ptr addrspace(3) %43, i64 -2
  %51 = getelementptr float, ptr addrspace(3) %43, i64 -1
  %52 = mul nuw nsw i32 %1042, 36
  %53 = add nsw i32 %1048, -40
  %54 = add nsw i32 %53, %.zext
  %55 = add nsw i32 %54, %52
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:84 within `kernel_mul_mask!`
; ┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br i1 %40, label %pass257, label %L814.1.critedge

L814.1.critedge:                                  ; preds = %L428.preheader
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br label %L814.1

L814.1:                                           ; preds = %pass257, %L814.1.critedge
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br i1 %40, label %pass257.2, label %L814.3.critedge

pass257.2:                                        ; preds = %L814.1
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %56 = load float, ptr addrspace(3) %44, align 4
             %57 = load float, ptr addrspace(3) %45, align 4
             %58 = load float, ptr addrspace(3) %46, align 4
             %59 = load float, ptr addrspace(3) %47, align 4
             %60 = load float, ptr addrspace(3) %48, align 4
             %61 = load float, ptr addrspace(3) %49, align 4
             %62 = load float, ptr addrspace(3) %50, align 4
             %63 = load float, ptr addrspace(3) %51, align 4
             %64 = zext nneg i32 %1049 to i64
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
              %65 = getelementptr float, ptr addrspace(3) @shmem39, i64 %64
              %66 = load float, ptr addrspace(3) %65, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %67 = fmul float %56, %66
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %68 = getelementptr float, ptr addrspace(3) %65, i64 36
              %69 = load float, ptr addrspace(3) %68, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %70 = fmul float %57, %69
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %71 = getelementptr float, ptr addrspace(3) %65, i64 72
              %72 = load float, ptr addrspace(3) %71, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %73 = fmul float %58, %72
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %74 = getelementptr float, ptr addrspace(3) %65, i64 108
              %75 = load float, ptr addrspace(3) %74, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %76 = fmul float %59, %75
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %77 = getelementptr float, ptr addrspace(3) %65, i64 144
              %78 = load float, ptr addrspace(3) %77, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %79 = fmul float %60, %78
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %80 = getelementptr float, ptr addrspace(3) %65, i64 180
              %81 = load float, ptr addrspace(3) %80, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %82 = fmul float %61, %81
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %83 = getelementptr float, ptr addrspace(3) %65, i64 216
              %84 = load float, ptr addrspace(3) %83, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %85 = fmul float %62, %84
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %86 = getelementptr float, ptr addrspace(3) %65, i64 252
              %87 = load float, ptr addrspace(3) %86, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %88 = fmul float %63, %87
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
                    %89 = fadd float %67, %70
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %90 = fadd float %89, %73
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %91 = fadd float %90, %76
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %92 = fadd float %91, %79
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %93 = fadd float %92, %82
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %94 = fadd float %93, %85
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %95 = fadd float %94, %88
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %96 = sext i32 %55 to i64
           %97 = getelementptr float, ptr addrspace(3) @shmem42, i64 %96
           %98 = getelementptr float, ptr addrspace(3) %97, i64 4
           store float %95, ptr addrspace(3) %98, align 4
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
              %99 = getelementptr float, ptr addrspace(3) %65, i64 4
              %100 = load float, ptr addrspace(3) %99, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %101 = fmul float %56, %100
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %102 = getelementptr float, ptr addrspace(3) %65, i64 40
              %103 = load float, ptr addrspace(3) %102, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %104 = fmul float %57, %103
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %105 = getelementptr float, ptr addrspace(3) %65, i64 76
              %106 = load float, ptr addrspace(3) %105, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %107 = fmul float %58, %106
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %108 = getelementptr float, ptr addrspace(3) %65, i64 112
              %109 = load float, ptr addrspace(3) %108, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %110 = fmul float %59, %109
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %111 = getelementptr float, ptr addrspace(3) %65, i64 148
              %112 = load float, ptr addrspace(3) %111, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %113 = fmul float %60, %112
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %114 = getelementptr float, ptr addrspace(3) %65, i64 184
              %115 = load float, ptr addrspace(3) %114, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %116 = fmul float %61, %115
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %117 = getelementptr float, ptr addrspace(3) %65, i64 220
              %118 = load float, ptr addrspace(3) %117, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %119 = fmul float %62, %118
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %120 = getelementptr float, ptr addrspace(3) %65, i64 256
              %121 = load float, ptr addrspace(3) %120, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %122 = fmul float %63, %121
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
                    %123 = fadd float %101, %104
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %124 = fadd float %123, %107
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %125 = fadd float %124, %110
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %126 = fadd float %125, %113
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %127 = fadd float %126, %116
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %128 = fadd float %127, %119
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %129 = fadd float %128, %122
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %130 = getelementptr float, ptr addrspace(3) %97, i64 8
           store float %129, ptr addrspace(3) %130, align 4
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
              %131 = getelementptr float, ptr addrspace(3) %65, i64 8
              %132 = load float, ptr addrspace(3) %131, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %133 = fmul float %56, %132
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %134 = getelementptr float, ptr addrspace(3) %65, i64 44
              %135 = load float, ptr addrspace(3) %134, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %136 = fmul float %57, %135
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %137 = getelementptr float, ptr addrspace(3) %65, i64 80
              %138 = load float, ptr addrspace(3) %137, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %139 = fmul float %58, %138
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %140 = getelementptr float, ptr addrspace(3) %65, i64 116
              %141 = load float, ptr addrspace(3) %140, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %142 = fmul float %59, %141
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %143 = getelementptr float, ptr addrspace(3) %65, i64 152
              %144 = load float, ptr addrspace(3) %143, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %145 = fmul float %60, %144
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %146 = getelementptr float, ptr addrspace(3) %65, i64 188
              %147 = load float, ptr addrspace(3) %146, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %148 = fmul float %61, %147
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %149 = getelementptr float, ptr addrspace(3) %65, i64 224
              %150 = load float, ptr addrspace(3) %149, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %151 = fmul float %62, %150
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %152 = getelementptr float, ptr addrspace(3) %65, i64 260
              %153 = load float, ptr addrspace(3) %152, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %154 = fmul float %63, %153
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
                    %155 = fadd float %133, %136
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %156 = fadd float %155, %139
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %157 = fadd float %156, %142
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %158 = fadd float %157, %145
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %159 = fadd float %158, %148
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %160 = fadd float %159, %151
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %161 = fadd float %160, %154
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %162 = getelementptr float, ptr addrspace(3) %97, i64 12
           store float %161, ptr addrspace(3) %162, align 4
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
              %163 = getelementptr float, ptr addrspace(3) %65, i64 12
              %164 = load float, ptr addrspace(3) %163, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %165 = fmul float %56, %164
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %166 = getelementptr float, ptr addrspace(3) %65, i64 48
              %167 = load float, ptr addrspace(3) %166, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %168 = fmul float %57, %167
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %169 = getelementptr float, ptr addrspace(3) %65, i64 84
              %170 = load float, ptr addrspace(3) %169, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %171 = fmul float %58, %170
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %172 = getelementptr float, ptr addrspace(3) %65, i64 120
              %173 = load float, ptr addrspace(3) %172, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %174 = fmul float %59, %173
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %175 = getelementptr float, ptr addrspace(3) %65, i64 156
              %176 = load float, ptr addrspace(3) %175, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %177 = fmul float %60, %176
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %178 = getelementptr float, ptr addrspace(3) %65, i64 192
              %179 = load float, ptr addrspace(3) %178, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %180 = fmul float %61, %179
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %181 = getelementptr float, ptr addrspace(3) %65, i64 228
              %182 = load float, ptr addrspace(3) %181, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %183 = fmul float %62, %182
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %184 = getelementptr float, ptr addrspace(3) %65, i64 264
              %185 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %186 = fmul float %63, %185
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
                    %187 = fadd float %165, %168
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %188 = fadd float %187, %171
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %189 = fadd float %188, %174
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %190 = fadd float %189, %177
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %191 = fadd float %190, %180
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %192 = fadd float %191, %183
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %193 = fadd float %192, %186
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %194 = getelementptr float, ptr addrspace(3) %97, i64 16
           store float %193, ptr addrspace(3) %194, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
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
             %195 = load float, ptr addrspace(3) %44, align 4
             %196 = load float, ptr addrspace(3) %45, align 4
             %197 = load float, ptr addrspace(3) %46, align 4
             %198 = load float, ptr addrspace(3) %47, align 4
             %199 = load float, ptr addrspace(3) %48, align 4
             %200 = load float, ptr addrspace(3) %49, align 4
             %201 = load float, ptr addrspace(3) %50, align 4
             %202 = load float, ptr addrspace(3) %51, align 4
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
              %203 = load float, ptr addrspace(3) %65, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %204 = fmul float %195, %203
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %205 = load float, ptr addrspace(3) %68, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %206 = fmul float %196, %205
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %207 = load float, ptr addrspace(3) %71, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %208 = fmul float %197, %207
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %209 = load float, ptr addrspace(3) %74, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %210 = fmul float %198, %209
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %211 = load float, ptr addrspace(3) %77, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %212 = fmul float %199, %211
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %213 = load float, ptr addrspace(3) %80, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %214 = fmul float %200, %213
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %215 = load float, ptr addrspace(3) %83, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %216 = fmul float %201, %215
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %217 = load float, ptr addrspace(3) %86, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %218 = fmul float %202, %217
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
                    %219 = fadd float %204, %206
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %220 = fadd float %219, %208
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %221 = fadd float %220, %210
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %222 = fadd float %221, %212
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %223 = fadd float %222, %214
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %224 = fadd float %223, %216
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %225 = fadd float %224, %218
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %225, ptr addrspace(3) %98, align 4
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
              %226 = load float, ptr addrspace(3) %99, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %227 = fmul float %195, %226
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %228 = load float, ptr addrspace(3) %102, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %229 = fmul float %196, %228
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %230 = load float, ptr addrspace(3) %105, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %231 = fmul float %197, %230
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %232 = load float, ptr addrspace(3) %108, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %233 = fmul float %198, %232
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %234 = load float, ptr addrspace(3) %111, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %235 = fmul float %199, %234
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %236 = load float, ptr addrspace(3) %114, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %237 = fmul float %200, %236
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %238 = load float, ptr addrspace(3) %117, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %239 = fmul float %201, %238
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %240 = load float, ptr addrspace(3) %120, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %241 = fmul float %202, %240
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
                    %242 = fadd float %227, %229
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %243 = fadd float %242, %231
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %244 = fadd float %243, %233
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %245 = fadd float %244, %235
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %246 = fadd float %245, %237
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %247 = fadd float %246, %239
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
           store float %248, ptr addrspace(3) %130, align 4
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
              %249 = load float, ptr addrspace(3) %131, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %250 = fmul float %195, %249
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %251 = load float, ptr addrspace(3) %134, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %252 = fmul float %196, %251
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %253 = load float, ptr addrspace(3) %137, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %254 = fmul float %197, %253
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %255 = load float, ptr addrspace(3) %140, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %256 = fmul float %198, %255
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %257 = load float, ptr addrspace(3) %143, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %258 = fmul float %199, %257
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %259 = load float, ptr addrspace(3) %146, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %260 = fmul float %200, %259
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %261 = load float, ptr addrspace(3) %149, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %262 = fmul float %201, %261
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %263 = load float, ptr addrspace(3) %152, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %264 = fmul float %202, %263
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
                    %265 = fadd float %250, %252
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %266 = fadd float %265, %254
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %267 = fadd float %266, %256
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %268 = fadd float %267, %258
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %269 = fadd float %268, %260
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %270 = fadd float %269, %262
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %271 = fadd float %270, %264
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %271, ptr addrspace(3) %162, align 4
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
              %272 = load float, ptr addrspace(3) %163, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %273 = fmul float %195, %272
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %274 = load float, ptr addrspace(3) %166, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %275 = fmul float %196, %274
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %276 = load float, ptr addrspace(3) %169, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %277 = fmul float %197, %276
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %278 = load float, ptr addrspace(3) %172, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %279 = fmul float %198, %278
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %280 = load float, ptr addrspace(3) %175, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %281 = fmul float %199, %280
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %282 = load float, ptr addrspace(3) %178, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %283 = fmul float %200, %282
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %284 = load float, ptr addrspace(3) %181, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %285 = fmul float %201, %284
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %286 = load float, ptr addrspace(3) %184, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %287 = fmul float %202, %286
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
                    %288 = fadd float %273, %275
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %289 = fadd float %288, %277
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %290 = fadd float %289, %279
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %291 = fadd float %290, %281
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %292 = fadd float %291, %283
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %293 = fadd float %292, %285
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %294 = fadd float %293, %287
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %294, ptr addrspace(3) %194, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    br label %L814.3

L814.3.critedge:                                  ; preds = %L814.1
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br label %L814.3

L814.3:                                           ; preds = %L814.3.critedge, %pass257.2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br i1 %40, label %pass257.4, label %L814.5.critedge

pass257.4:                                        ; preds = %L814.3
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %295 = load float, ptr addrspace(3) %44, align 4
             %296 = load float, ptr addrspace(3) %45, align 4
             %297 = load float, ptr addrspace(3) %46, align 4
             %298 = load float, ptr addrspace(3) %47, align 4
             %299 = load float, ptr addrspace(3) %48, align 4
             %300 = load float, ptr addrspace(3) %49, align 4
             %301 = load float, ptr addrspace(3) %50, align 4
             %302 = load float, ptr addrspace(3) %51, align 4
             %303 = zext nneg i32 %1049 to i64
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
              %304 = getelementptr float, ptr addrspace(3) @shmem39, i64 %303
              %305 = load float, ptr addrspace(3) %304, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %306 = fmul float %295, %305
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %307 = getelementptr float, ptr addrspace(3) %304, i64 36
              %308 = load float, ptr addrspace(3) %307, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %309 = fmul float %296, %308
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %310 = getelementptr float, ptr addrspace(3) %304, i64 72
              %311 = load float, ptr addrspace(3) %310, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %312 = fmul float %297, %311
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %313 = getelementptr float, ptr addrspace(3) %304, i64 108
              %314 = load float, ptr addrspace(3) %313, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %315 = fmul float %298, %314
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %316 = getelementptr float, ptr addrspace(3) %304, i64 144
              %317 = load float, ptr addrspace(3) %316, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %318 = fmul float %299, %317
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %319 = getelementptr float, ptr addrspace(3) %304, i64 180
              %320 = load float, ptr addrspace(3) %319, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %321 = fmul float %300, %320
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %322 = getelementptr float, ptr addrspace(3) %304, i64 216
              %323 = load float, ptr addrspace(3) %322, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %324 = fmul float %301, %323
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %325 = getelementptr float, ptr addrspace(3) %304, i64 252
              %326 = load float, ptr addrspace(3) %325, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %327 = fmul float %302, %326
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
                    %328 = fadd float %306, %309
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %329 = fadd float %328, %312
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %330 = fadd float %329, %315
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %331 = fadd float %330, %318
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %332 = fadd float %331, %321
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %333 = fadd float %332, %324
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %334 = fadd float %333, %327
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %335 = sext i32 %55 to i64
           %336 = getelementptr float, ptr addrspace(3) @shmem42, i64 %335
           %337 = getelementptr float, ptr addrspace(3) %336, i64 4
           store float %334, ptr addrspace(3) %337, align 4
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
              %338 = getelementptr float, ptr addrspace(3) %304, i64 4
              %339 = load float, ptr addrspace(3) %338, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %340 = fmul float %295, %339
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %341 = getelementptr float, ptr addrspace(3) %304, i64 40
              %342 = load float, ptr addrspace(3) %341, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %343 = fmul float %296, %342
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %344 = getelementptr float, ptr addrspace(3) %304, i64 76
              %345 = load float, ptr addrspace(3) %344, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %346 = fmul float %297, %345
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %347 = getelementptr float, ptr addrspace(3) %304, i64 112
              %348 = load float, ptr addrspace(3) %347, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %349 = fmul float %298, %348
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %350 = getelementptr float, ptr addrspace(3) %304, i64 148
              %351 = load float, ptr addrspace(3) %350, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %352 = fmul float %299, %351
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %353 = getelementptr float, ptr addrspace(3) %304, i64 184
              %354 = load float, ptr addrspace(3) %353, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %355 = fmul float %300, %354
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %356 = getelementptr float, ptr addrspace(3) %304, i64 220
              %357 = load float, ptr addrspace(3) %356, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %358 = fmul float %301, %357
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %359 = getelementptr float, ptr addrspace(3) %304, i64 256
              %360 = load float, ptr addrspace(3) %359, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %361 = fmul float %302, %360
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
                    %362 = fadd float %340, %343
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %363 = fadd float %362, %346
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %364 = fadd float %363, %349
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %365 = fadd float %364, %352
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %366 = fadd float %365, %355
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %367 = fadd float %366, %358
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %368 = fadd float %367, %361
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %369 = getelementptr float, ptr addrspace(3) %336, i64 8
           store float %368, ptr addrspace(3) %369, align 4
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
              %370 = getelementptr float, ptr addrspace(3) %304, i64 8
              %371 = load float, ptr addrspace(3) %370, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %372 = fmul float %295, %371
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %373 = getelementptr float, ptr addrspace(3) %304, i64 44
              %374 = load float, ptr addrspace(3) %373, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %375 = fmul float %296, %374
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %376 = getelementptr float, ptr addrspace(3) %304, i64 80
              %377 = load float, ptr addrspace(3) %376, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %378 = fmul float %297, %377
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %379 = getelementptr float, ptr addrspace(3) %304, i64 116
              %380 = load float, ptr addrspace(3) %379, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %381 = fmul float %298, %380
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %382 = getelementptr float, ptr addrspace(3) %304, i64 152
              %383 = load float, ptr addrspace(3) %382, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %384 = fmul float %299, %383
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %385 = getelementptr float, ptr addrspace(3) %304, i64 188
              %386 = load float, ptr addrspace(3) %385, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %387 = fmul float %300, %386
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %388 = getelementptr float, ptr addrspace(3) %304, i64 224
              %389 = load float, ptr addrspace(3) %388, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %390 = fmul float %301, %389
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %391 = getelementptr float, ptr addrspace(3) %304, i64 260
              %392 = load float, ptr addrspace(3) %391, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %393 = fmul float %302, %392
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
                    %394 = fadd float %372, %375
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %395 = fadd float %394, %378
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %396 = fadd float %395, %381
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %397 = fadd float %396, %384
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %398 = fadd float %397, %387
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %399 = fadd float %398, %390
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %400 = fadd float %399, %393
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %401 = getelementptr float, ptr addrspace(3) %336, i64 12
           store float %400, ptr addrspace(3) %401, align 4
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
              %402 = getelementptr float, ptr addrspace(3) %304, i64 12
              %403 = load float, ptr addrspace(3) %402, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %404 = fmul float %295, %403
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %405 = getelementptr float, ptr addrspace(3) %304, i64 48
              %406 = load float, ptr addrspace(3) %405, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %407 = fmul float %296, %406
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %408 = getelementptr float, ptr addrspace(3) %304, i64 84
              %409 = load float, ptr addrspace(3) %408, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %410 = fmul float %297, %409
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %411 = getelementptr float, ptr addrspace(3) %304, i64 120
              %412 = load float, ptr addrspace(3) %411, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %413 = fmul float %298, %412
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %414 = getelementptr float, ptr addrspace(3) %304, i64 156
              %415 = load float, ptr addrspace(3) %414, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %416 = fmul float %299, %415
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %417 = getelementptr float, ptr addrspace(3) %304, i64 192
              %418 = load float, ptr addrspace(3) %417, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %419 = fmul float %300, %418
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %420 = getelementptr float, ptr addrspace(3) %304, i64 228
              %421 = load float, ptr addrspace(3) %420, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %422 = fmul float %301, %421
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %423 = getelementptr float, ptr addrspace(3) %304, i64 264
              %424 = load float, ptr addrspace(3) %423, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %425 = fmul float %302, %424
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
                    %426 = fadd float %404, %407
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %427 = fadd float %426, %410
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %428 = fadd float %427, %413
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %429 = fadd float %428, %416
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %430 = fadd float %429, %419
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %431 = fadd float %430, %422
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %432 = fadd float %431, %425
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %433 = getelementptr float, ptr addrspace(3) %336, i64 16
           store float %432, ptr addrspace(3) %433, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
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
             %434 = load float, ptr addrspace(3) %44, align 4
             %435 = load float, ptr addrspace(3) %45, align 4
             %436 = load float, ptr addrspace(3) %46, align 4
             %437 = load float, ptr addrspace(3) %47, align 4
             %438 = load float, ptr addrspace(3) %48, align 4
             %439 = load float, ptr addrspace(3) %49, align 4
             %440 = load float, ptr addrspace(3) %50, align 4
             %441 = load float, ptr addrspace(3) %51, align 4
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
              %442 = load float, ptr addrspace(3) %304, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %443 = fmul float %434, %442
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %444 = load float, ptr addrspace(3) %307, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %445 = fmul float %435, %444
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %446 = load float, ptr addrspace(3) %310, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %447 = fmul float %436, %446
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %448 = load float, ptr addrspace(3) %313, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %449 = fmul float %437, %448
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %450 = load float, ptr addrspace(3) %316, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %451 = fmul float %438, %450
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %452 = load float, ptr addrspace(3) %319, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %453 = fmul float %439, %452
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %454 = load float, ptr addrspace(3) %322, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %455 = fmul float %440, %454
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %456 = load float, ptr addrspace(3) %325, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %457 = fmul float %441, %456
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
                    %458 = fadd float %443, %445
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %459 = fadd float %458, %447
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %460 = fadd float %459, %449
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %461 = fadd float %460, %451
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %462 = fadd float %461, %453
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %463 = fadd float %462, %455
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %464 = fadd float %463, %457
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %464, ptr addrspace(3) %337, align 4
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
              %465 = load float, ptr addrspace(3) %338, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %466 = fmul float %434, %465
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %467 = load float, ptr addrspace(3) %341, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %468 = fmul float %435, %467
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %469 = load float, ptr addrspace(3) %344, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %470 = fmul float %436, %469
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %471 = load float, ptr addrspace(3) %347, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %472 = fmul float %437, %471
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %473 = load float, ptr addrspace(3) %350, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %474 = fmul float %438, %473
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %475 = load float, ptr addrspace(3) %353, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %476 = fmul float %439, %475
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %477 = load float, ptr addrspace(3) %356, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %478 = fmul float %440, %477
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %479 = load float, ptr addrspace(3) %359, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %480 = fmul float %441, %479
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
                    %481 = fadd float %466, %468
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %482 = fadd float %481, %470
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %483 = fadd float %482, %472
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %484 = fadd float %483, %474
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %485 = fadd float %484, %476
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %486 = fadd float %485, %478
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %487 = fadd float %486, %480
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %487, ptr addrspace(3) %369, align 4
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
              %488 = load float, ptr addrspace(3) %370, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %489 = fmul float %434, %488
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %490 = load float, ptr addrspace(3) %373, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %491 = fmul float %435, %490
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %492 = load float, ptr addrspace(3) %376, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %493 = fmul float %436, %492
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %494 = load float, ptr addrspace(3) %379, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %495 = fmul float %437, %494
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %496 = load float, ptr addrspace(3) %382, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %497 = fmul float %438, %496
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %498 = load float, ptr addrspace(3) %385, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %499 = fmul float %439, %498
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %500 = load float, ptr addrspace(3) %388, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %501 = fmul float %440, %500
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %502 = load float, ptr addrspace(3) %391, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %503 = fmul float %441, %502
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
                    %504 = fadd float %489, %491
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %505 = fadd float %504, %493
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %506 = fadd float %505, %495
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %507 = fadd float %506, %497
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %508 = fadd float %507, %499
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %509 = fadd float %508, %501
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %510 = fadd float %509, %503
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %510, ptr addrspace(3) %401, align 4
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
              %511 = load float, ptr addrspace(3) %402, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %512 = fmul float %434, %511
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %513 = load float, ptr addrspace(3) %405, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %514 = fmul float %435, %513
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %515 = load float, ptr addrspace(3) %408, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %516 = fmul float %436, %515
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %517 = load float, ptr addrspace(3) %411, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %518 = fmul float %437, %517
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %519 = load float, ptr addrspace(3) %414, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %520 = fmul float %438, %519
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %521 = load float, ptr addrspace(3) %417, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %522 = fmul float %439, %521
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %523 = load float, ptr addrspace(3) %420, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %524 = fmul float %440, %523
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %525 = load float, ptr addrspace(3) %423, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %526 = fmul float %441, %525
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
                    %527 = fadd float %512, %514
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %528 = fadd float %527, %516
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %529 = fadd float %528, %518
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %530 = fadd float %529, %520
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %531 = fadd float %530, %522
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %532 = fadd float %531, %524
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %533 = fadd float %532, %526
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %533, ptr addrspace(3) %433, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    br label %L814.5

L814.5.critedge:                                  ; preds = %L814.3
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br label %L814.5

L814.5:                                           ; preds = %L814.5.critedge, %pass257.4
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br i1 %40, label %pass257.6, label %L814.7.critedge

pass257.6:                                        ; preds = %L814.5
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %534 = load float, ptr addrspace(3) %44, align 4
             %535 = load float, ptr addrspace(3) %45, align 4
             %536 = load float, ptr addrspace(3) %46, align 4
             %537 = load float, ptr addrspace(3) %47, align 4
             %538 = load float, ptr addrspace(3) %48, align 4
             %539 = load float, ptr addrspace(3) %49, align 4
             %540 = load float, ptr addrspace(3) %50, align 4
             %541 = load float, ptr addrspace(3) %51, align 4
             %542 = zext nneg i32 %1049 to i64
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
              %543 = getelementptr float, ptr addrspace(3) @shmem39, i64 %542
              %544 = load float, ptr addrspace(3) %543, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %545 = fmul float %534, %544
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %546 = getelementptr float, ptr addrspace(3) %543, i64 36
              %547 = load float, ptr addrspace(3) %546, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %548 = fmul float %535, %547
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %549 = getelementptr float, ptr addrspace(3) %543, i64 72
              %550 = load float, ptr addrspace(3) %549, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %551 = fmul float %536, %550
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %552 = getelementptr float, ptr addrspace(3) %543, i64 108
              %553 = load float, ptr addrspace(3) %552, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %554 = fmul float %537, %553
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %555 = getelementptr float, ptr addrspace(3) %543, i64 144
              %556 = load float, ptr addrspace(3) %555, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %557 = fmul float %538, %556
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %558 = getelementptr float, ptr addrspace(3) %543, i64 180
              %559 = load float, ptr addrspace(3) %558, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %560 = fmul float %539, %559
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %561 = getelementptr float, ptr addrspace(3) %543, i64 216
              %562 = load float, ptr addrspace(3) %561, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %563 = fmul float %540, %562
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %564 = getelementptr float, ptr addrspace(3) %543, i64 252
              %565 = load float, ptr addrspace(3) %564, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %566 = fmul float %541, %565
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
                    %567 = fadd float %545, %548
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %568 = fadd float %567, %551
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %569 = fadd float %568, %554
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %570 = fadd float %569, %557
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %571 = fadd float %570, %560
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %572 = fadd float %571, %563
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %573 = fadd float %572, %566
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %574 = sext i32 %55 to i64
           %575 = getelementptr float, ptr addrspace(3) @shmem42, i64 %574
           %576 = getelementptr float, ptr addrspace(3) %575, i64 4
           store float %573, ptr addrspace(3) %576, align 4
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
              %577 = getelementptr float, ptr addrspace(3) %543, i64 4
              %578 = load float, ptr addrspace(3) %577, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %579 = fmul float %534, %578
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %580 = getelementptr float, ptr addrspace(3) %543, i64 40
              %581 = load float, ptr addrspace(3) %580, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %582 = fmul float %535, %581
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %583 = getelementptr float, ptr addrspace(3) %543, i64 76
              %584 = load float, ptr addrspace(3) %583, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %585 = fmul float %536, %584
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %586 = getelementptr float, ptr addrspace(3) %543, i64 112
              %587 = load float, ptr addrspace(3) %586, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %588 = fmul float %537, %587
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %589 = getelementptr float, ptr addrspace(3) %543, i64 148
              %590 = load float, ptr addrspace(3) %589, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %591 = fmul float %538, %590
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %592 = getelementptr float, ptr addrspace(3) %543, i64 184
              %593 = load float, ptr addrspace(3) %592, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %594 = fmul float %539, %593
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %595 = getelementptr float, ptr addrspace(3) %543, i64 220
              %596 = load float, ptr addrspace(3) %595, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %597 = fmul float %540, %596
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %598 = getelementptr float, ptr addrspace(3) %543, i64 256
              %599 = load float, ptr addrspace(3) %598, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %600 = fmul float %541, %599
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
                    %601 = fadd float %579, %582
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %602 = fadd float %601, %585
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %603 = fadd float %602, %588
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %604 = fadd float %603, %591
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %605 = fadd float %604, %594
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %606 = fadd float %605, %597
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %607 = fadd float %606, %600
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %608 = getelementptr float, ptr addrspace(3) %575, i64 8
           store float %607, ptr addrspace(3) %608, align 4
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
              %609 = getelementptr float, ptr addrspace(3) %543, i64 8
              %610 = load float, ptr addrspace(3) %609, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %611 = fmul float %534, %610
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %612 = getelementptr float, ptr addrspace(3) %543, i64 44
              %613 = load float, ptr addrspace(3) %612, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %614 = fmul float %535, %613
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %615 = getelementptr float, ptr addrspace(3) %543, i64 80
              %616 = load float, ptr addrspace(3) %615, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %617 = fmul float %536, %616
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %618 = getelementptr float, ptr addrspace(3) %543, i64 116
              %619 = load float, ptr addrspace(3) %618, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %620 = fmul float %537, %619
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %621 = getelementptr float, ptr addrspace(3) %543, i64 152
              %622 = load float, ptr addrspace(3) %621, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %623 = fmul float %538, %622
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %624 = getelementptr float, ptr addrspace(3) %543, i64 188
              %625 = load float, ptr addrspace(3) %624, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %626 = fmul float %539, %625
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %627 = getelementptr float, ptr addrspace(3) %543, i64 224
              %628 = load float, ptr addrspace(3) %627, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %629 = fmul float %540, %628
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %630 = getelementptr float, ptr addrspace(3) %543, i64 260
              %631 = load float, ptr addrspace(3) %630, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %632 = fmul float %541, %631
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
                    %633 = fadd float %611, %614
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %634 = fadd float %633, %617
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %635 = fadd float %634, %620
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %636 = fadd float %635, %623
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %637 = fadd float %636, %626
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %638 = fadd float %637, %629
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %639 = fadd float %638, %632
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %640 = getelementptr float, ptr addrspace(3) %575, i64 12
           store float %639, ptr addrspace(3) %640, align 4
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
              %641 = getelementptr float, ptr addrspace(3) %543, i64 12
              %642 = load float, ptr addrspace(3) %641, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %643 = fmul float %534, %642
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %644 = getelementptr float, ptr addrspace(3) %543, i64 48
              %645 = load float, ptr addrspace(3) %644, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %646 = fmul float %535, %645
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %647 = getelementptr float, ptr addrspace(3) %543, i64 84
              %648 = load float, ptr addrspace(3) %647, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %649 = fmul float %536, %648
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %650 = getelementptr float, ptr addrspace(3) %543, i64 120
              %651 = load float, ptr addrspace(3) %650, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %652 = fmul float %537, %651
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %653 = getelementptr float, ptr addrspace(3) %543, i64 156
              %654 = load float, ptr addrspace(3) %653, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %655 = fmul float %538, %654
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %656 = getelementptr float, ptr addrspace(3) %543, i64 192
              %657 = load float, ptr addrspace(3) %656, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %658 = fmul float %539, %657
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %659 = getelementptr float, ptr addrspace(3) %543, i64 228
              %660 = load float, ptr addrspace(3) %659, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %661 = fmul float %540, %660
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %662 = getelementptr float, ptr addrspace(3) %543, i64 264
              %663 = load float, ptr addrspace(3) %662, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %664 = fmul float %541, %663
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
                    %665 = fadd float %643, %646
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %666 = fadd float %665, %649
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %667 = fadd float %666, %652
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %668 = fadd float %667, %655
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %669 = fadd float %668, %658
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %670 = fadd float %669, %661
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %671 = fadd float %670, %664
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %672 = getelementptr float, ptr addrspace(3) %575, i64 16
           store float %671, ptr addrspace(3) %672, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
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
             %673 = load float, ptr addrspace(3) %44, align 4
             %674 = load float, ptr addrspace(3) %45, align 4
             %675 = load float, ptr addrspace(3) %46, align 4
             %676 = load float, ptr addrspace(3) %47, align 4
             %677 = load float, ptr addrspace(3) %48, align 4
             %678 = load float, ptr addrspace(3) %49, align 4
             %679 = load float, ptr addrspace(3) %50, align 4
             %680 = load float, ptr addrspace(3) %51, align 4
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
              %681 = load float, ptr addrspace(3) %543, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %682 = fmul float %673, %681
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %683 = load float, ptr addrspace(3) %546, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %684 = fmul float %674, %683
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %685 = load float, ptr addrspace(3) %549, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %686 = fmul float %675, %685
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %687 = load float, ptr addrspace(3) %552, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %688 = fmul float %676, %687
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %689 = load float, ptr addrspace(3) %555, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %690 = fmul float %677, %689
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %691 = load float, ptr addrspace(3) %558, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %692 = fmul float %678, %691
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %693 = load float, ptr addrspace(3) %561, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %694 = fmul float %679, %693
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %695 = load float, ptr addrspace(3) %564, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %696 = fmul float %680, %695
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
                    %697 = fadd float %682, %684
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %698 = fadd float %697, %686
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %699 = fadd float %698, %688
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %700 = fadd float %699, %690
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %701 = fadd float %700, %692
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %702 = fadd float %701, %694
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %703 = fadd float %702, %696
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %703, ptr addrspace(3) %576, align 4
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
              %704 = load float, ptr addrspace(3) %577, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %705 = fmul float %673, %704
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %706 = load float, ptr addrspace(3) %580, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %707 = fmul float %674, %706
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %708 = load float, ptr addrspace(3) %583, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %709 = fmul float %675, %708
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %710 = load float, ptr addrspace(3) %586, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %711 = fmul float %676, %710
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %712 = load float, ptr addrspace(3) %589, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %713 = fmul float %677, %712
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %714 = load float, ptr addrspace(3) %592, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %715 = fmul float %678, %714
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %716 = load float, ptr addrspace(3) %595, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %717 = fmul float %679, %716
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %718 = load float, ptr addrspace(3) %598, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %719 = fmul float %680, %718
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
                    %720 = fadd float %705, %707
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %721 = fadd float %720, %709
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %722 = fadd float %721, %711
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %723 = fadd float %722, %713
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %724 = fadd float %723, %715
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %725 = fadd float %724, %717
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %726 = fadd float %725, %719
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %726, ptr addrspace(3) %608, align 4
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
              %727 = load float, ptr addrspace(3) %609, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %728 = fmul float %673, %727
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %729 = load float, ptr addrspace(3) %612, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %730 = fmul float %674, %729
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %731 = load float, ptr addrspace(3) %615, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %732 = fmul float %675, %731
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %733 = load float, ptr addrspace(3) %618, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %734 = fmul float %676, %733
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %735 = load float, ptr addrspace(3) %621, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %736 = fmul float %677, %735
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %737 = load float, ptr addrspace(3) %624, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %738 = fmul float %678, %737
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %739 = load float, ptr addrspace(3) %627, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %740 = fmul float %679, %739
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %741 = load float, ptr addrspace(3) %630, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %742 = fmul float %680, %741
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
                    %743 = fadd float %728, %730
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %744 = fadd float %743, %732
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %745 = fadd float %744, %734
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %746 = fadd float %745, %736
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %747 = fadd float %746, %738
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %748 = fadd float %747, %740
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %749 = fadd float %748, %742
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %749, ptr addrspace(3) %640, align 4
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
              %750 = load float, ptr addrspace(3) %641, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %751 = fmul float %673, %750
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %752 = load float, ptr addrspace(3) %644, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %753 = fmul float %674, %752
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %754 = load float, ptr addrspace(3) %647, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %755 = fmul float %675, %754
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %756 = load float, ptr addrspace(3) %650, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %757 = fmul float %676, %756
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %758 = load float, ptr addrspace(3) %653, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %759 = fmul float %677, %758
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %760 = load float, ptr addrspace(3) %656, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %761 = fmul float %678, %760
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %762 = load float, ptr addrspace(3) %659, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %763 = fmul float %679, %762
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %764 = load float, ptr addrspace(3) %662, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %765 = fmul float %680, %764
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
                    %766 = fadd float %751, %753
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %767 = fadd float %766, %755
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %768 = fadd float %767, %757
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %769 = fadd float %768, %759
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %770 = fadd float %769, %761
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %771 = fadd float %770, %763
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %772 = fadd float %771, %765
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %772, ptr addrspace(3) %672, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    br label %L814.7

L814.7.critedge:                                  ; preds = %L814.5
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br label %L814.7

L814.7:                                           ; preds = %L814.7.critedge, %pass257.6
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br i1 %40, label %pass257.8, label %L814.9.critedge

pass257.8:                                        ; preds = %L814.7
; ││ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `batch_op!`
; ││┌ @ ntuple.jl:71 within `ntuple`
; │││┌ @ ntuple.jl:74 within `macro expansion`
; ││││┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:160 within `#batch_op!##0`
; │││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:508 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; ││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││││││┌ @ none within `pointerref`
; ││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
             %773 = load float, ptr addrspace(3) %44, align 4
             %774 = load float, ptr addrspace(3) %45, align 4
             %775 = load float, ptr addrspace(3) %46, align 4
             %776 = load float, ptr addrspace(3) %47, align 4
             %777 = load float, ptr addrspace(3) %48, align 4
             %778 = load float, ptr addrspace(3) %49, align 4
             %779 = load float, ptr addrspace(3) %50, align 4
             %780 = load float, ptr addrspace(3) %51, align 4
             %781 = zext nneg i32 %1049 to i64
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
              %782 = getelementptr float, ptr addrspace(3) @shmem39, i64 %781
              %783 = load float, ptr addrspace(3) %782, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %784 = fmul float %773, %783
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %785 = getelementptr float, ptr addrspace(3) %782, i64 36
              %786 = load float, ptr addrspace(3) %785, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %787 = fmul float %774, %786
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %788 = getelementptr float, ptr addrspace(3) %782, i64 72
              %789 = load float, ptr addrspace(3) %788, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %790 = fmul float %775, %789
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %791 = getelementptr float, ptr addrspace(3) %782, i64 108
              %792 = load float, ptr addrspace(3) %791, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %793 = fmul float %776, %792
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %794 = getelementptr float, ptr addrspace(3) %782, i64 144
              %795 = load float, ptr addrspace(3) %794, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %796 = fmul float %777, %795
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %797 = getelementptr float, ptr addrspace(3) %782, i64 180
              %798 = load float, ptr addrspace(3) %797, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %799 = fmul float %778, %798
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %800 = getelementptr float, ptr addrspace(3) %782, i64 216
              %801 = load float, ptr addrspace(3) %800, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %802 = fmul float %779, %801
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %803 = getelementptr float, ptr addrspace(3) %782, i64 252
              %804 = load float, ptr addrspace(3) %803, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %805 = fmul float %780, %804
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
                    %806 = fadd float %784, %787
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %807 = fadd float %806, %790
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %808 = fadd float %807, %793
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %809 = fadd float %808, %796
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %810 = fadd float %809, %799
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %811 = fadd float %810, %802
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %812 = fadd float %811, %805
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %813 = sext i32 %55 to i64
           %814 = getelementptr float, ptr addrspace(3) @shmem42, i64 %813
           %815 = getelementptr float, ptr addrspace(3) %814, i64 4
           store float %812, ptr addrspace(3) %815, align 4
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
              %816 = getelementptr float, ptr addrspace(3) %782, i64 4
              %817 = load float, ptr addrspace(3) %816, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %818 = fmul float %773, %817
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %819 = getelementptr float, ptr addrspace(3) %782, i64 40
              %820 = load float, ptr addrspace(3) %819, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %821 = fmul float %774, %820
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %822 = getelementptr float, ptr addrspace(3) %782, i64 76
              %823 = load float, ptr addrspace(3) %822, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %824 = fmul float %775, %823
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %825 = getelementptr float, ptr addrspace(3) %782, i64 112
              %826 = load float, ptr addrspace(3) %825, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %827 = fmul float %776, %826
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %828 = getelementptr float, ptr addrspace(3) %782, i64 148
              %829 = load float, ptr addrspace(3) %828, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %830 = fmul float %777, %829
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %831 = getelementptr float, ptr addrspace(3) %782, i64 184
              %832 = load float, ptr addrspace(3) %831, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %833 = fmul float %778, %832
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %834 = getelementptr float, ptr addrspace(3) %782, i64 220
              %835 = load float, ptr addrspace(3) %834, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %836 = fmul float %779, %835
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %837 = getelementptr float, ptr addrspace(3) %782, i64 256
              %838 = load float, ptr addrspace(3) %837, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %839 = fmul float %780, %838
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
                    %840 = fadd float %818, %821
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %841 = fadd float %840, %824
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %842 = fadd float %841, %827
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %843 = fadd float %842, %830
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %844 = fadd float %843, %833
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %845 = fadd float %844, %836
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %846 = fadd float %845, %839
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %847 = getelementptr float, ptr addrspace(3) %814, i64 8
           store float %846, ptr addrspace(3) %847, align 4
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
              %848 = getelementptr float, ptr addrspace(3) %782, i64 8
              %849 = load float, ptr addrspace(3) %848, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %850 = fmul float %773, %849
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %851 = getelementptr float, ptr addrspace(3) %782, i64 44
              %852 = load float, ptr addrspace(3) %851, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %853 = fmul float %774, %852
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %854 = getelementptr float, ptr addrspace(3) %782, i64 80
              %855 = load float, ptr addrspace(3) %854, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %856 = fmul float %775, %855
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %857 = getelementptr float, ptr addrspace(3) %782, i64 116
              %858 = load float, ptr addrspace(3) %857, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %859 = fmul float %776, %858
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %860 = getelementptr float, ptr addrspace(3) %782, i64 152
              %861 = load float, ptr addrspace(3) %860, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %862 = fmul float %777, %861
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %863 = getelementptr float, ptr addrspace(3) %782, i64 188
              %864 = load float, ptr addrspace(3) %863, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %865 = fmul float %778, %864
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %866 = getelementptr float, ptr addrspace(3) %782, i64 224
              %867 = load float, ptr addrspace(3) %866, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %868 = fmul float %779, %867
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %869 = getelementptr float, ptr addrspace(3) %782, i64 260
              %870 = load float, ptr addrspace(3) %869, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %871 = fmul float %780, %870
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
                    %872 = fadd float %850, %853
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %873 = fadd float %872, %856
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %874 = fadd float %873, %859
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %875 = fadd float %874, %862
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %876 = fadd float %875, %865
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %877 = fadd float %876, %868
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %878 = fadd float %877, %871
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %879 = getelementptr float, ptr addrspace(3) %814, i64 12
           store float %878, ptr addrspace(3) %879, align 4
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
              %880 = getelementptr float, ptr addrspace(3) %782, i64 12
              %881 = load float, ptr addrspace(3) %880, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %882 = fmul float %773, %881
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %883 = getelementptr float, ptr addrspace(3) %782, i64 48
              %884 = load float, ptr addrspace(3) %883, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %885 = fmul float %774, %884
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %886 = getelementptr float, ptr addrspace(3) %782, i64 84
              %887 = load float, ptr addrspace(3) %886, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %888 = fmul float %775, %887
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %889 = getelementptr float, ptr addrspace(3) %782, i64 120
              %890 = load float, ptr addrspace(3) %889, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %891 = fmul float %776, %890
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %892 = getelementptr float, ptr addrspace(3) %782, i64 156
              %893 = load float, ptr addrspace(3) %892, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %894 = fmul float %777, %893
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %895 = getelementptr float, ptr addrspace(3) %782, i64 192
              %896 = load float, ptr addrspace(3) %895, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %897 = fmul float %778, %896
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %898 = getelementptr float, ptr addrspace(3) %782, i64 228
              %899 = load float, ptr addrspace(3) %898, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %900 = fmul float %779, %899
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %901 = getelementptr float, ptr addrspace(3) %782, i64 264
              %902 = load float, ptr addrspace(3) %901, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %903 = fmul float %780, %902
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
                    %904 = fadd float %882, %885
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %905 = fadd float %904, %888
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %906 = fadd float %905, %891
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %907 = fadd float %906, %894
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %908 = fadd float %907, %897
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %909 = fadd float %908, %900
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %910 = fadd float %909, %903
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %911 = getelementptr float, ptr addrspace(3) %814, i64 16
           store float %910, ptr addrspace(3) %911, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
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
             %912 = load float, ptr addrspace(3) %44, align 4
             %913 = load float, ptr addrspace(3) %45, align 4
             %914 = load float, ptr addrspace(3) %46, align 4
             %915 = load float, ptr addrspace(3) %47, align 4
             %916 = load float, ptr addrspace(3) %48, align 4
             %917 = load float, ptr addrspace(3) %49, align 4
             %918 = load float, ptr addrspace(3) %50, align 4
             %919 = load float, ptr addrspace(3) %51, align 4
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
              %920 = load float, ptr addrspace(3) %782, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %921 = fmul float %912, %920
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %922 = load float, ptr addrspace(3) %785, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %923 = fmul float %913, %922
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %924 = load float, ptr addrspace(3) %788, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %925 = fmul float %914, %924
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %926 = load float, ptr addrspace(3) %791, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %927 = fmul float %915, %926
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %928 = load float, ptr addrspace(3) %794, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %929 = fmul float %916, %928
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %930 = load float, ptr addrspace(3) %797, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %931 = fmul float %917, %930
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %932 = load float, ptr addrspace(3) %800, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %933 = fmul float %918, %932
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %934 = load float, ptr addrspace(3) %803, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %935 = fmul float %919, %934
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
                    %936 = fadd float %921, %923
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %937 = fadd float %936, %925
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %938 = fadd float %937, %927
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %939 = fadd float %938, %929
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %940 = fadd float %939, %931
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %941 = fadd float %940, %933
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %942 = fadd float %941, %935
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %942, ptr addrspace(3) %815, align 4
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
              %943 = load float, ptr addrspace(3) %816, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %944 = fmul float %912, %943
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %945 = load float, ptr addrspace(3) %819, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %946 = fmul float %913, %945
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %947 = load float, ptr addrspace(3) %822, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %948 = fmul float %914, %947
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %949 = load float, ptr addrspace(3) %825, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %950 = fmul float %915, %949
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %951 = load float, ptr addrspace(3) %828, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %952 = fmul float %916, %951
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %953 = load float, ptr addrspace(3) %831, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %954 = fmul float %917, %953
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %955 = load float, ptr addrspace(3) %834, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %956 = fmul float %918, %955
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %957 = load float, ptr addrspace(3) %837, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %958 = fmul float %919, %957
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
                    %959 = fadd float %944, %946
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %960 = fadd float %959, %948
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %961 = fadd float %960, %950
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %962 = fadd float %961, %952
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %963 = fadd float %962, %954
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %964 = fadd float %963, %956
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %965 = fadd float %964, %958
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %965, ptr addrspace(3) %847, align 4
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
              %966 = load float, ptr addrspace(3) %848, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %967 = fmul float %912, %966
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %968 = load float, ptr addrspace(3) %851, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %969 = fmul float %913, %968
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %970 = load float, ptr addrspace(3) %854, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %971 = fmul float %914, %970
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %972 = load float, ptr addrspace(3) %857, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %973 = fmul float %915, %972
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %974 = load float, ptr addrspace(3) %860, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %975 = fmul float %916, %974
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %976 = load float, ptr addrspace(3) %863, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %977 = fmul float %917, %976
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %978 = load float, ptr addrspace(3) %866, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %979 = fmul float %918, %978
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %980 = load float, ptr addrspace(3) %869, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %981 = fmul float %919, %980
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
                    %982 = fadd float %967, %969
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %983 = fadd float %982, %971
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %984 = fadd float %983, %973
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %985 = fadd float %984, %975
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %986 = fadd float %985, %977
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %987 = fadd float %986, %979
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %988 = fadd float %987, %981
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %988, ptr addrspace(3) %879, align 4
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
              %989 = load float, ptr addrspace(3) %880, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %990 = fmul float %912, %989
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %991 = load float, ptr addrspace(3) %883, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %992 = fmul float %913, %991
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %993 = load float, ptr addrspace(3) %886, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %994 = fmul float %914, %993
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %995 = load float, ptr addrspace(3) %889, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %996 = fmul float %915, %995
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %997 = load float, ptr addrspace(3) %892, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %998 = fmul float %916, %997
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %999 = load float, ptr addrspace(3) %895, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1000 = fmul float %917, %999
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1001 = load float, ptr addrspace(3) %898, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1002 = fmul float %918, %1001
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1003 = load float, ptr addrspace(3) %901, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1004 = fmul float %919, %1003
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
                    %1005 = fadd float %990, %992
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1006 = fadd float %1005, %994
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1007 = fadd float %1006, %996
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1008 = fadd float %1007, %998
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1009 = fadd float %1008, %1000
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1010 = fadd float %1009, %1002
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1011 = fadd float %1010, %1004
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1011, ptr addrspace(3) %911, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    br label %L814.9

L814.9.critedge:                                  ; preds = %L814.7
    call void asm sideeffect "", "~{memory}"() #2
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/src/operations.jl:155 within `batch_op!`
    br label %L814.9

L814.9:                                           ; preds = %L814.9.critedge, %pass257.8
; │└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    call void asm sideeffect "", "~{memory}"() #2
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:90 within `kernel_mul_mask!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:60 within `sync_warp` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:60
   br label %L826

L826:                                             ; preds = %L814.9, %L422
   call void @llvm.nvvm.bar.warp.sync(i32 -1)
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:93 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1360 within `dual_to_interm_transfer!`
; │┌ @ int.jl:87 within `+`
    %1012 = add i32 %1044, %1069
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1361 within `dual_to_interm_transfer!`
; │┌ @ int.jl:87 within `+`
    %1013 = add i32 %1012, %.zext
    %.not40 = icmp sgt i32 %1013, %"N::Int32"
    %1014 = icmp ugt i32 %1042, 4
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1363 within `dual_to_interm_transfer!`
   %or.cond11 = select i1 %.not40, i1 true, i1 %1014
   br i1 %or.cond11, label %pass232, label %L906.preheader

L1069:                                            ; preds = %pass243, %pass232
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:94 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1429 within `intermediate_layout_write!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %1015 = add nuw nsw i32 %1161, 31
     %1016 = add nuw nsw i32 %1161, 30
     %1017 = lshr i32 %1016, 4
; ││└
; ││┌ @ int.jl:87 within `+`
     %1018 = add i32 %1044, %1017
; ││└
; ││┌ @ int.jl:520 within `<=`
     %1019 = icmp ugt i32 %1161, 481
     %.not43.1 = icmp sgt i32 %1018, %"N::Int32"
; ││└
    %or.cond13.1 = select i1 %1019, i1 true, i1 %.not43.1
    %.not44.1 = icmp ugt i32 %1015, %1162
    %or.cond91 = select i1 %or.cond13.1, i1 true, i1 %.not44.1
    br i1 %or.cond91, label %L1069.1, label %pass243.1

pass243.1:                                        ; preds = %L1069
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1445
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1020 = add i32 %1164, %1015
; ││││││││└
          %1021 = sext i32 %1020 to i64
          %1022 = getelementptr inbounds float, ptr addrspace(1) %"M_out::CuDeviceArray.fca.0.extract", i64 %1021
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1023 = sext i32 %1056 to i64
          %1024 = getelementptr float, ptr addrspace(3) @shmem39, i64 %1023
          %1025 = getelementptr float, ptr addrspace(3) %1024, i64 33
          %1026 = load float, ptr addrspace(3) %1025, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %1026, ptr addrspace(1) %1022, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L1069.1

L1069.1:                                          ; preds = %pass243.1, %L1069
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:96 within `kernel_mul_mask!`
  ret void

pass13:                                           ; preds = %conversion
  %"A_global::CuDeviceArray.fca.0.extract" = extractvalue { ptr addrspace(1), i64, [2 x i64], i64 } %"A_global::CuDeviceArray", 0
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:64 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:625 within `shared_matrix_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:71 within `threadIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:46 within `threadIdx_x`
; │││┌ @ int.jl:87 within `+`
      %1027 = add nuw nsw i32 %0, 1
; │└└└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:626 within `shared_matrix_load!`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %1028 = and i32 %1027, 31
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not = icmp eq i32 %1028, 0
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:634 within `shared_matrix_load!`
; │┌ @ int.jl:86 within `-`
    %1029 = add nsw i32 %1028, -1
    %1030 = select i1 %.not, i32 31, i32 %1029
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:639 within `shared_matrix_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; ││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; │││││┌ @ none within `pointerref`
; ││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
         %1031 = zext nneg i32 %1030 to i64
         %1032 = getelementptr inbounds float, ptr addrspace(1) %"A_global::CuDeviceArray.fca.0.extract", i64 %1031
         %1033 = load float, ptr addrspace(1) %1032, align 4
; │└└└└└└
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││┌ @ none within `pointerset`
; ││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
         %1034 = getelementptr inbounds float, ptr addrspace(3) @shmem, i64 %1031
         store float %1033, ptr addrspace(3) %1034, align 4
; └└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:69 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:19 within `get_shmem_elems`
; │┌ @ int.jl:301 within `div`
    br label %pass30

pass30:                                           ; preds = %pass13, %conversion.pass30_crit_edge
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %.pre-phi2 = phi i32 [ %.pre1, %conversion.pass30_crit_edge ], [ %1028, %pass13 ]
; │└└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:27 within `get_shmem_elems`
; │┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:85 within `blockIdx`
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:56 within `blockIdx_x`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `_index`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/indexing.jl:7 within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
       %1035 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; │└└└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:28 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not25 = icmp eq i32 %.pre-phi2, 0
; ││└
; ││┌ @ essentials.jl:799 within `ifelse`
     %1036 = select i1 %.not25, i32 32, i32 %.pre-phi2
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:30 within `get_shmem_elems`
; │┌ @ int.jl:86 within `-`
    %1037 = add nsw i32 %1036, -1
; │└
; │┌ @ int.jl:301 within `div`
    %1038 = lshr i32 %1037, 3
    %.zext = and i32 %1038, 31
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:31 within `get_shmem_elems`
; │┌ @ int.jl:88 within `*`
    %1039 = lshr i32 %0, 3
    %1040 = and i32 %1039, 124
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:32 within `get_shmem_elems`
; │┌ @ operators.jl:885 within `mod1`
; ││┌ @ int.jl:287 within `mod`
; │││┌ @ int.jl:86 within `-`
      %1041 = and i32 %1036, 7
; ││└└
; ││┌ @ promotion.jl:487 within `==` @ promotion.jl:637
     %.not26 = icmp eq i32 %1041, 0
; ││└
; ││┌ @ essentials.jl:799 within `ifelse`
     %1042 = select i1 %.not26, i32 8, i32 %1041
; │└└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:36 within `get_shmem_elems`
; │┌ @ int.jl:88 within `*`
    %1043 = shl i32 %1035, 5
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:30 within `get_shmem_elems`
; │┌ @ int.jl:87 within `+`
    %1044 = or disjoint i32 %1043, 1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:31 within `get_shmem_elems`
; │┌ @ int.jl:87 within `+`
    %1045 = add i32 %1044, %1040
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:36 within `get_shmem_elems`
; │┌ @ int.jl:87 within `+`
    %1046 = add i32 %1045, %.zext
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:74 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:29 within `DualAccessMatrix`
; │┌ @ int.jl:301 within `div`
    %1047 = lshr i32 %0, 5
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:34 within `DualAccessMatrix`
; │┌ @ int.jl:88 within `*`
    %1048 = mul nuw nsw i32 %1047, 284
; │└
; │┌ @ int.jl:86 within `-`
    %1049 = add nuw nsw i32 %.zext, %1048
; └└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:78 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1186 within `intermediate_layout_load!`
; │┌ @ int.jl:88 within `*`
    %1050 = shl nuw nsw i32 %1047, 7
; │└
; │┌ @ int.jl:87 within `+`
    %1051 = or disjoint i32 %1050, 1
    %1052 = add nuw nsw i32 %1051, %1036
    %1053 = add nuw nsw i32 %1050, 128
    %1054 = shl i32 %1035, 10
    %1055 = add i32 %1054, -1
    %1056 = add nuw nsw i32 %1037, %1048
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1191 within `intermediate_layout_load!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %1057 = add nsw i32 %1052, -1
     %1058 = add nsw i32 %1052, -2
; ││└
; ││┌ @ int.jl:301 within `div`
     %1059 = lshr i32 %1058, 5
     %.zext52 = and i32 %1059, 2047
; ││└
; ││┌ @ int.jl:87 within `+`
     %1060 = add i32 %1044, %.zext52
; ││└
; ││┌ @ int.jl:520 within `<=`
     %1061 = icmp ugt i32 %1058, 1023
     %.not28 = icmp sgt i32 %1060, %"N::Int32"
; ││└
    %or.cond = select i1 %1061, i1 true, i1 %.not28
    %.not29 = icmp ugt i32 %1057, %1053
    %or.cond78 = select i1 %or.cond, i1 true, i1 %.not29
    br i1 %or.cond78, label %L277, label %pass107

pass107:                                          ; preds = %pass30
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1207
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1062 = zext nneg i32 %1056 to i64
          %1063 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %1062
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1064 = add i32 %1055, %1057
; ││││││││└
          %1065 = sext i32 %1064 to i64
          %1066 = getelementptr inbounds float, ptr addrspace(1) %"M_in::CuDeviceArray.fca.0.extract", i64 %1065
          %1067 = load float, ptr addrspace(1) %1066, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %1067, ptr addrspace(3) %1063, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L277

pass143:                                          ; preds = %pass107.3, %L277.2
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:79 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1278 within `interm_to_dual_transfer!`
; │┌ @ int.jl:87 within `+`
    %1068 = add nuw nsw i32 %.zext, 1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1280 within `interm_to_dual_transfer!`
; │┌ @ int.jl:88 within `*`
    %1069 = shl nuw nsw i32 %1047, 2
; │└
; │┌ @ int.jl:87 within `+`
    %1070 = add i32 %1069, %1043
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1281 within `interm_to_dual_transfer!`
; │┌ @ int.jl:87 within `+`
    %1071 = add i32 %1070, %1068
    %.not33 = icmp sgt i32 %1071, %"N::Int32"
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1283 within `interm_to_dual_transfer!`
   br i1 %.not33, label %L422, label %L355.preheader

L355.preheader:                                   ; preds = %pass143
   %1072 = shl nuw nsw i32 %.zext, 5
   %1073 = shl nuw nsw i32 %1042, 2
   %1074 = add nsw i32 %1073, -4
   %1075 = add nsw i32 %1074, %1072
   %1076 = shl nuw nsw i32 %1047, 3
   %1077 = add nsw i32 %1076, -1
   %1078 = add nsw i32 %1077, %1042
   %1079 = mul nsw i32 %1078, 36
   %1080 = sub nsw i32 %.zext, %1069
   %1081 = add nsw i32 %1080, -4
   %1082 = add nsw i32 %1081, %1079
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1284 within `interm_to_dual_transfer!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc66 = trunc i32 %1075 to i8
     %1083 = sdiv i8 %.lhs.trunc66, 32
     %.sext67 = sext i8 %1083 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1084 = add nsw i32 %1075, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1085 = add nsw i32 %1084, %.sext67
; ││││││││└
          %1086 = sext i32 %1085 to i64
          %1087 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %1086
          %1088 = load float, ptr addrspace(3) %1087, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1089 = add nsw i32 %1080, %1079
; ││││││││└
          %1090 = sext i32 %1089 to i64
          %1091 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1090
          store float %1088, ptr addrspace(3) %1091, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1092 = or disjoint i32 %1075, 1
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc68 = trunc i32 %1092 to i8
     %1093 = sdiv i8 %.lhs.trunc68, 32
     %.sext69 = sext i8 %1093 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1094 = add nsw i32 %1092, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1095 = add nsw i32 %1094, %.sext69
; ││││││││└
          %1096 = sext i32 %1095 to i64
          %1097 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %1096
          %1098 = load float, ptr addrspace(3) %1097, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1099 = sext i32 %1082 to i64
          %1100 = getelementptr float, ptr addrspace(3) @shmem39, i64 %1099
          %1101 = getelementptr float, ptr addrspace(3) %1100, i64 8
          store float %1098, ptr addrspace(3) %1101, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1102 = or disjoint i32 %1075, 2
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc70 = trunc i32 %1102 to i8
     %1103 = sdiv i8 %.lhs.trunc70, 32
     %.sext71 = sext i8 %1103 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1104 = add nsw i32 %1102, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1105 = add nsw i32 %1104, %.sext71
; ││││││││└
          %1106 = sext i32 %1105 to i64
          %1107 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %1106
          %1108 = load float, ptr addrspace(3) %1107, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1109 = getelementptr float, ptr addrspace(3) %1100, i64 12
          store float %1108, ptr addrspace(3) %1109, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1110 = or disjoint i32 %1075, 3
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc72 = trunc i32 %1110 to i8
     %1111 = sdiv i8 %.lhs.trunc72, 32
     %.sext73 = sext i8 %1111 to i32
; ││└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1112 = add nsw i32 %1110, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1113 = add nsw i32 %1112, %.sext73
; ││││││││└
          %1114 = sext i32 %1113 to i64
          %1115 = getelementptr inbounds float, ptr addrspace(3) @shmem42, i64 %1114
          %1116 = load float, ptr addrspace(3) %1115, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1117 = getelementptr float, ptr addrspace(3) %1100, i64 16
          store float %1116, ptr addrspace(3) %1117, align 4
; └└└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:81 within `kernel_mul_mask!`
; ┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/intrinsics/synchronization.jl:17 within `sync_threads`
   br label %L422

L906.preheader:                                   ; preds = %L826
   %1118 = shl nuw nsw i32 %.zext, 4
   %1119 = shl nuw nsw i32 %1042, 2
   %1120 = add nsw i32 %1119, -4
   %1121 = add nsw i32 %1120, %1118
   %1122 = mul nuw nsw i32 %1042, 36
   %1123 = add nsw i32 %1048, -40
   %1124 = add nsw i32 %1123, %.zext
   %1125 = add nsw i32 %1124, %1122
; └
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:93 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1364 within `dual_to_interm_transfer!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc59 = trunc i32 %1121 to i8
     %1126 = sdiv i8 %.lhs.trunc59, 32
     %.sext = sext i8 %1126 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1127 = sext i32 %1125 to i64
          %1128 = getelementptr float, ptr addrspace(3) @shmem42, i64 %1127
          %1129 = getelementptr float, ptr addrspace(3) %1128, i64 4
          %1130 = load float, ptr addrspace(3) %1129, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1131 = add nsw i32 %1121, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1132 = add nsw i32 %1131, %.sext
; ││││││││└
          %1133 = sext i32 %1132 to i64
          %1134 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1133
          store float %1130, ptr addrspace(3) %1134, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1135 = or disjoint i32 %1121, 1
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc60 = trunc i32 %1135 to i8
     %1136 = sdiv i8 %.lhs.trunc60, 32
     %.sext61 = sext i8 %1136 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1137 = getelementptr float, ptr addrspace(3) %1128, i64 8
          %1138 = load float, ptr addrspace(3) %1137, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1139 = add nsw i32 %1135, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1140 = add nsw i32 %1139, %.sext61
; ││││││││└
          %1141 = sext i32 %1140 to i64
          %1142 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1141
          store float %1138, ptr addrspace(3) %1142, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1143 = or disjoint i32 %1121, 2
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc62 = trunc i32 %1143 to i8
     %1144 = sdiv i8 %.lhs.trunc62, 32
     %.sext63 = sext i8 %1144 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1145 = getelementptr float, ptr addrspace(3) %1128, i64 12
          %1146 = load float, ptr addrspace(3) %1145, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1147 = add nsw i32 %1143, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1148 = add nsw i32 %1147, %.sext63
; ││││││││└
          %1149 = sext i32 %1148 to i64
          %1150 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1149
          store float %1146, ptr addrspace(3) %1150, align 4
; ││└└└└└└
; ││┌ @ int.jl:86 within `-`
     %1151 = or disjoint i32 %1121, 3
; ││└
; ││┌ @ int.jl:301 within `div`
     %.lhs.trunc64 = trunc i32 %1151 to i8
     %1152 = sdiv i8 %.lhs.trunc64, 32
     %.sext65 = sext i8 %1152 to i32
; ││└
; ││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1153 = getelementptr float, ptr addrspace(3) %1128, i64 16
          %1154 = load float, ptr addrspace(3) %1153, align 4
; ││└└└└└└
; ││┌ @ operators.jl:642 within `+` @ int.jl:87
     %1155 = add nsw i32 %1151, %1048
; ││└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1156 = add nsw i32 %1155, %.sext65
; ││││││││└
          %1157 = sext i32 %1156 to i64
          %1158 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1157
          store float %1154, ptr addrspace(3) %1158, align 4
; └└└└└└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:94 within `kernel_mul_mask!`
; ┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1417 within `intermediate_layout_write!`
; │┌ @ int.jl:301 within `div`
    br label %pass232

pass232:                                          ; preds = %L906.preheader, %L826
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1424 within `intermediate_layout_write!`
; │┌ @ int.jl:88 within `*`
    %1159 = shl nuw nsw i32 %1047, 6
; │└
; │┌ @ int.jl:87 within `+`
    %1160 = or disjoint i32 %1159, 1
    %1161 = add nuw nsw i32 %1160, %1036
    %1162 = add nuw nsw i32 %1159, 64
    %1163 = shl i32 %1035, 9
    %1164 = add i32 %1163, -1
; │└
; │ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1429 within `intermediate_layout_write!`
; │┌ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion`
; ││┌ @ int.jl:86 within `-`
     %1165 = add nsw i32 %1161, -1
     %1166 = add nsw i32 %1161, -2
; ││└
; ││┌ @ int.jl:301 within `div`
     %1167 = sdiv i32 %1166, 16
; ││└
; ││┌ @ int.jl:87 within `+`
     %1168 = add i32 %1044, %1167
; ││└
; ││┌ @ int.jl:520 within `<=`
     %1169 = icmp ugt i32 %1161, 513
     %.not43 = icmp sgt i32 %1168, %"N::Int32"
; ││└
    %or.cond13 = select i1 %1169, i1 true, i1 %.not43
    %.not44 = icmp ugt i32 %1165, %1162
    %or.cond90 = select i1 %or.cond13, i1 true, i1 %.not44
    br i1 %or.cond90, label %L1069, label %pass243

pass243:                                          ; preds = %pass232
; ││ @ /home/zelda/sy440/.julia/packages/KernelAbstractions/ecO4B/src/extras/loopinfo.jl:31 within `macro expansion` @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:1445
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
; ││││││││┌ @ int.jl:86 within `-`
           %1170 = add i32 %1164, %1165
; ││││││││└
          %1171 = sext i32 %1170 to i64
          %1172 = getelementptr inbounds float, ptr addrspace(1) %"M_out::CuDeviceArray.fca.0.extract", i64 %1171
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175 within `getindex`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││┌ @ none within `pointerref`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          %1173 = zext nneg i32 %1056 to i64
          %1174 = getelementptr inbounds float, ptr addrspace(3) @shmem39, i64 %1173
          %1175 = load float, ptr addrspace(3) %1174, align 4
; ││└└└└└└
; ││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177 within `setindex!`
; │││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; │││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; ││││││┌ @ none within `pointerset`
; │││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
          store float %1175, ptr addrspace(1) %1172, align 4
; ││││└└└└
; ││││ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:137 within `arrayset`
      br label %L1069

pass257:                                          ; preds = %L428.preheader
; └└└└
;  @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:84 within `kernel_mul_mask!`
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
             %1176 = load float, ptr addrspace(3) %44, align 4
             %1177 = load float, ptr addrspace(3) %45, align 4
             %1178 = load float, ptr addrspace(3) %46, align 4
             %1179 = load float, ptr addrspace(3) %47, align 4
             %1180 = load float, ptr addrspace(3) %48, align 4
             %1181 = load float, ptr addrspace(3) %49, align 4
             %1182 = load float, ptr addrspace(3) %50, align 4
             %1183 = load float, ptr addrspace(3) %51, align 4
             %1184 = zext nneg i32 %1049 to i64
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
              %1185 = getelementptr float, ptr addrspace(3) @shmem39, i64 %1184
              %1186 = load float, ptr addrspace(3) %1185, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1187 = fmul float %1176, %1186
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1188 = getelementptr float, ptr addrspace(3) %1185, i64 36
              %1189 = load float, ptr addrspace(3) %1188, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1190 = fmul float %1177, %1189
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1191 = getelementptr float, ptr addrspace(3) %1185, i64 72
              %1192 = load float, ptr addrspace(3) %1191, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1193 = fmul float %1178, %1192
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1194 = getelementptr float, ptr addrspace(3) %1185, i64 108
              %1195 = load float, ptr addrspace(3) %1194, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1196 = fmul float %1179, %1195
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1197 = getelementptr float, ptr addrspace(3) %1185, i64 144
              %1198 = load float, ptr addrspace(3) %1197, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1199 = fmul float %1180, %1198
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1200 = getelementptr float, ptr addrspace(3) %1185, i64 180
              %1201 = load float, ptr addrspace(3) %1200, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1202 = fmul float %1181, %1201
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1203 = getelementptr float, ptr addrspace(3) %1185, i64 216
              %1204 = load float, ptr addrspace(3) %1203, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1205 = fmul float %1182, %1204
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1206 = getelementptr float, ptr addrspace(3) %1185, i64 252
              %1207 = load float, ptr addrspace(3) %1206, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1208 = fmul float %1183, %1207
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
                    %1209 = fadd float %1187, %1190
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1210 = fadd float %1209, %1193
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1211 = fadd float %1210, %1196
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1212 = fadd float %1211, %1199
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1213 = fadd float %1212, %1202
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1214 = fadd float %1213, %1205
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1215 = fadd float %1214, %1208
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %1216 = sext i32 %55 to i64
           %1217 = getelementptr float, ptr addrspace(3) @shmem42, i64 %1216
           %1218 = getelementptr float, ptr addrspace(3) %1217, i64 4
           store float %1215, ptr addrspace(3) %1218, align 4
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
              %1219 = getelementptr float, ptr addrspace(3) %1185, i64 4
              %1220 = load float, ptr addrspace(3) %1219, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1221 = fmul float %1176, %1220
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1222 = getelementptr float, ptr addrspace(3) %1185, i64 40
              %1223 = load float, ptr addrspace(3) %1222, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1224 = fmul float %1177, %1223
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1225 = getelementptr float, ptr addrspace(3) %1185, i64 76
              %1226 = load float, ptr addrspace(3) %1225, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1227 = fmul float %1178, %1226
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1228 = getelementptr float, ptr addrspace(3) %1185, i64 112
              %1229 = load float, ptr addrspace(3) %1228, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1230 = fmul float %1179, %1229
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1231 = getelementptr float, ptr addrspace(3) %1185, i64 148
              %1232 = load float, ptr addrspace(3) %1231, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1233 = fmul float %1180, %1232
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1234 = getelementptr float, ptr addrspace(3) %1185, i64 184
              %1235 = load float, ptr addrspace(3) %1234, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1236 = fmul float %1181, %1235
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1237 = getelementptr float, ptr addrspace(3) %1185, i64 220
              %1238 = load float, ptr addrspace(3) %1237, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1239 = fmul float %1182, %1238
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1240 = getelementptr float, ptr addrspace(3) %1185, i64 256
              %1241 = load float, ptr addrspace(3) %1240, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1242 = fmul float %1183, %1241
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
                    %1243 = fadd float %1221, %1224
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1244 = fadd float %1243, %1227
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1245 = fadd float %1244, %1230
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1246 = fadd float %1245, %1233
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1247 = fadd float %1246, %1236
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1248 = fadd float %1247, %1239
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1249 = fadd float %1248, %1242
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %1250 = getelementptr float, ptr addrspace(3) %1217, i64 8
           store float %1249, ptr addrspace(3) %1250, align 4
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
              %1251 = getelementptr float, ptr addrspace(3) %1185, i64 8
              %1252 = load float, ptr addrspace(3) %1251, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1253 = fmul float %1176, %1252
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1254 = getelementptr float, ptr addrspace(3) %1185, i64 44
              %1255 = load float, ptr addrspace(3) %1254, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1256 = fmul float %1177, %1255
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1257 = getelementptr float, ptr addrspace(3) %1185, i64 80
              %1258 = load float, ptr addrspace(3) %1257, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1259 = fmul float %1178, %1258
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1260 = getelementptr float, ptr addrspace(3) %1185, i64 116
              %1261 = load float, ptr addrspace(3) %1260, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1262 = fmul float %1179, %1261
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1263 = getelementptr float, ptr addrspace(3) %1185, i64 152
              %1264 = load float, ptr addrspace(3) %1263, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1265 = fmul float %1180, %1264
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1266 = getelementptr float, ptr addrspace(3) %1185, i64 188
              %1267 = load float, ptr addrspace(3) %1266, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1268 = fmul float %1181, %1267
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1269 = getelementptr float, ptr addrspace(3) %1185, i64 224
              %1270 = load float, ptr addrspace(3) %1269, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1271 = fmul float %1182, %1270
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1272 = getelementptr float, ptr addrspace(3) %1185, i64 260
              %1273 = load float, ptr addrspace(3) %1272, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1274 = fmul float %1183, %1273
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
                    %1275 = fadd float %1253, %1256
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1276 = fadd float %1275, %1259
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1277 = fadd float %1276, %1262
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1278 = fadd float %1277, %1265
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1279 = fadd float %1278, %1268
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1280 = fadd float %1279, %1271
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1281 = fadd float %1280, %1274
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %1282 = getelementptr float, ptr addrspace(3) %1217, i64 12
           store float %1281, ptr addrspace(3) %1282, align 4
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
              %1283 = getelementptr float, ptr addrspace(3) %1185, i64 12
              %1284 = load float, ptr addrspace(3) %1283, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1285 = fmul float %1176, %1284
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1286 = getelementptr float, ptr addrspace(3) %1185, i64 48
              %1287 = load float, ptr addrspace(3) %1286, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1288 = fmul float %1177, %1287
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1289 = getelementptr float, ptr addrspace(3) %1185, i64 84
              %1290 = load float, ptr addrspace(3) %1289, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1291 = fmul float %1178, %1290
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1292 = getelementptr float, ptr addrspace(3) %1185, i64 120
              %1293 = load float, ptr addrspace(3) %1292, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1294 = fmul float %1179, %1293
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1295 = getelementptr float, ptr addrspace(3) %1185, i64 156
              %1296 = load float, ptr addrspace(3) %1295, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1297 = fmul float %1180, %1296
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1298 = getelementptr float, ptr addrspace(3) %1185, i64 192
              %1299 = load float, ptr addrspace(3) %1298, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1300 = fmul float %1181, %1299
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1301 = getelementptr float, ptr addrspace(3) %1185, i64 228
              %1302 = load float, ptr addrspace(3) %1301, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1303 = fmul float %1182, %1302
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1304 = getelementptr float, ptr addrspace(3) %1185, i64 264
              %1305 = load float, ptr addrspace(3) %1304, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1306 = fmul float %1183, %1305
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
                    %1307 = fadd float %1285, %1288
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1308 = fadd float %1307, %1291
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1309 = fadd float %1308, %1294
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1310 = fadd float %1309, %1297
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1311 = fadd float %1310, %1300
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1312 = fadd float %1311, %1303
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1313 = fadd float %1312, %1306
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           %1314 = getelementptr float, ptr addrspace(3) %1217, i64 16
           store float %1313, ptr addrspace(3) %1314, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
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
             %1315 = load float, ptr addrspace(3) %44, align 4
             %1316 = load float, ptr addrspace(3) %45, align 4
             %1317 = load float, ptr addrspace(3) %46, align 4
             %1318 = load float, ptr addrspace(3) %47, align 4
             %1319 = load float, ptr addrspace(3) %48, align 4
             %1320 = load float, ptr addrspace(3) %49, align 4
             %1321 = load float, ptr addrspace(3) %50, align 4
             %1322 = load float, ptr addrspace(3) %51, align 4
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
              %1323 = load float, ptr addrspace(3) %1185, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1324 = fmul float %1315, %1323
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1325 = load float, ptr addrspace(3) %1188, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1326 = fmul float %1316, %1325
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1327 = load float, ptr addrspace(3) %1191, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1328 = fmul float %1317, %1327
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1329 = load float, ptr addrspace(3) %1194, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1330 = fmul float %1318, %1329
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1331 = load float, ptr addrspace(3) %1197, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1332 = fmul float %1319, %1331
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1333 = load float, ptr addrspace(3) %1200, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1334 = fmul float %1320, %1333
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1335 = load float, ptr addrspace(3) %1203, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1336 = fmul float %1321, %1335
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1337 = load float, ptr addrspace(3) %1206, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1338 = fmul float %1322, %1337
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
                    %1339 = fadd float %1324, %1326
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1340 = fadd float %1339, %1328
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1341 = fadd float %1340, %1330
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1342 = fadd float %1341, %1332
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1343 = fadd float %1342, %1334
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1344 = fadd float %1343, %1336
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1345 = fadd float %1344, %1338
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1345, ptr addrspace(3) %1218, align 4
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
              %1346 = load float, ptr addrspace(3) %1219, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1347 = fmul float %1315, %1346
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1348 = load float, ptr addrspace(3) %1222, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1349 = fmul float %1316, %1348
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1350 = load float, ptr addrspace(3) %1225, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1351 = fmul float %1317, %1350
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1352 = load float, ptr addrspace(3) %1228, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1353 = fmul float %1318, %1352
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1354 = load float, ptr addrspace(3) %1231, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1355 = fmul float %1319, %1354
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1356 = load float, ptr addrspace(3) %1234, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1357 = fmul float %1320, %1356
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1358 = load float, ptr addrspace(3) %1237, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1359 = fmul float %1321, %1358
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1360 = load float, ptr addrspace(3) %1240, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1361 = fmul float %1322, %1360
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
                    %1362 = fadd float %1347, %1349
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1363 = fadd float %1362, %1351
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1364 = fadd float %1363, %1353
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1365 = fadd float %1364, %1355
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1366 = fadd float %1365, %1357
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1367 = fadd float %1366, %1359
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1368 = fadd float %1367, %1361
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1368, ptr addrspace(3) %1250, align 4
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
              %1369 = load float, ptr addrspace(3) %1251, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1370 = fmul float %1315, %1369
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1371 = load float, ptr addrspace(3) %1254, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1372 = fmul float %1316, %1371
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1373 = load float, ptr addrspace(3) %1257, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1374 = fmul float %1317, %1373
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1375 = load float, ptr addrspace(3) %1260, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1376 = fmul float %1318, %1375
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1377 = load float, ptr addrspace(3) %1263, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1378 = fmul float %1319, %1377
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1379 = load float, ptr addrspace(3) %1266, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1380 = fmul float %1320, %1379
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1381 = load float, ptr addrspace(3) %1269, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1382 = fmul float %1321, %1381
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1383 = load float, ptr addrspace(3) %1272, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1384 = fmul float %1322, %1383
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
                    %1385 = fadd float %1370, %1372
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1386 = fadd float %1385, %1374
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1387 = fadd float %1386, %1376
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1388 = fadd float %1387, %1378
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1389 = fadd float %1388, %1380
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1390 = fadd float %1389, %1382
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1391 = fadd float %1390, %1384
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1391, ptr addrspace(3) %1282, align 4
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
              %1392 = load float, ptr addrspace(3) %1283, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1393 = fmul float %1315, %1392
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1394 = load float, ptr addrspace(3) %1286, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1395 = fmul float %1316, %1394
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1396 = load float, ptr addrspace(3) %1289, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1397 = fmul float %1317, %1396
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1398 = load float, ptr addrspace(3) %1292, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1399 = fmul float %1318, %1398
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1400 = load float, ptr addrspace(3) %1295, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1401 = fmul float %1319, %1400
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1402 = load float, ptr addrspace(3) %1298, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1403 = fmul float %1320, %1402
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1404 = load float, ptr addrspace(3) %1301, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1405 = fmul float %1321, %1404
; ││││││└
; ││││││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:86 within `getindex` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:175
; │││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:90 within `arrayref`
; ││││││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:96 within `arrayref_bits`
; │││││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:87 within `unsafe_load`
; ││││││││││┌ @ none within `pointerref`
; │││││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
              %1406 = load float, ptr addrspace(3) %1304, align 4
; ││││││└└└└└└
; ││││││┌ @ float.jl:497 within `*`
         %1407 = fmul float %1322, %1406
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
                    %1408 = fadd float %1393, %1395
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:601 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1409 = fadd float %1408, %1397
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:602 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1410 = fadd float %1409, %1399
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:603 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1411 = fadd float %1410, %1401
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:604 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1412 = fadd float %1411, %1403
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:605 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1413 = fadd float %1412, %1405
; │││││││││││││││└└└
; │││││││││││││││ @ operators.jl:606 within `afoldl`
; │││││││││││││││┌ @ reduce.jl:78 within `BottomRF`
; ││││││││││││││││┌ @ reduce.jl:19 within `add_sum`
; │││││││││││││││││┌ @ float.jl:495 within `+`
                    %1414 = fadd float %1413, %1407
; │││└└└└└└└└└└└└└└└
; │││┌ @ /scratch/sy440/BatchedKernels.jl/src/memory.jl:97 within `setindex!` @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:177
; ││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:135 within `arrayset`
; │││││┌ @ /home/zelda/sy440/.julia/packages/CUDA/bjncr/src/device/array.jl:142 within `arrayset_bits`
; ││││││┌ @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/pointer.jl:90 within `unsafe_store!`
; │││││││┌ @ none within `pointerset`
; ││││││││┌ @ none within `macro expansion` @ /home/zelda/sy440/.julia/packages/LLVM/upRII/src/interop/base.jl:39
           store float %1414, ptr addrspace(3) %1314, align 4
; │└└└└└└└└
; │┌ @ /scratch/sy440/BatchedKernels.jl/benchmarking/studies/comparison_repeated_mul_mask_vs_defrag/repeated_mul_mask.jl:8 within `opaque_barrier`
    br label %L814.1
; └└
}
