@inline function _load_mat(::Val{false}, M)
    return M
end

@inline function _load_mat(::Val{true}, M)
    return M'
end

"""
Matrix multiplication kernel for the case where one warp handles multiple matrices.
"""
@inline function kernel_matmul!(
    Cs,
    As,
    Bs,
    ::Val{A_adj},
    ::Val{B_adj},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
    ::Val{mode},
) where {D,nthreads,A_adj,B_adj,mode}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_3, As, Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D), Val(nthreads), N)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        # Create dual-access matrices
        A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
        B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
        C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)

        # Perform operation with optional adjoints
        A_mat = _load_mat(Val(A_adj), A)
        B_mat = _load_mat(Val(B_adj), B)
        batch_op!(*, C, A_mat, B_mat, d, Val(D))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, shmem_3, Val(D), Val(nthreads), N)
    intermediate_layout_write!(
        Cs, shmem_1, Val(D), Val(nthreads), N, Val(mode)
    )

    return nothing
end

"""
Overloaded kernel called without A_adj, B_adj arguments, defaulting these to false
"""
@inline function kernel_matmul!(
    Cs, As, Bs, ::Val{D}, ::Val{nthreads}, N::Int32, ::Val{V}, ::Val{mode}
) where {D,nthreads,V,mode}
    return kernel_matmul!(
        Cs, As, Bs, Val(false), Val(false), Val(D), Val(nthreads), N, Val(V), Val(mode)
    )
end

@inline function kernel_matmul!(
    Cs,
    As,
    Bs,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D1,D2,D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_3, As, Val(D1), Val(D2), Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N)

    # Load B
    intermediate_layout_load!(shmem_3, Bs, Val(D2), Val(D1), Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D2), Val(D1), Val(D), Val(nthreads), N)

    # Create dual-access matrices
    A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
    B = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
    C = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        batch_op!(*, C, A, B, d, Val(D1), Val(D2), Val(D))
    end

    # Store C
    dual_to_interm_transfer!(shmem_1, C, Val(D1), Val(D1), Val(D), Val(nthreads), N)
    intermediate_layout_write!(Cs, shmem_1, Val(D1), Val(D1), Val(D), Val(nthreads), N)

    return nothing
end

@inline function kernel_trig_matmul!(
    C, A, L, ::Val{D1}, ::Val{D2}, ::Val{D}, ::Val{nthreads}, N::Int32,
) where {D1,D2,D,nthreads}
    # shape(A) = (D1,D2)
    # shape(L) = (D2,D2)

    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))
    M1 = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
    M2 = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)
    M3 = DualAccessMatrix(shmem_3, Val(D), warp_matrix_id)

    # Load M1 <- A
    intermediate_layout_load!(shmem_3, A, Val(D1), Val(D2), Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_1, shmem_3, Val(D1), Val(D2), Val(D), Val(nthreads), N)

    # Load M2 <- L
    intermediate_layout_load!(shmem_3, L, Val(D2), Val(D2), Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_2, shmem_3, Val(D2), Val(D2), Val(D), Val(nthreads), N)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        batch_op!(*, M3, M1, LowerTriangular(M2), d, Val(D1), Val(D2), Val(D))
    end

    dual_to_interm_transfer!(shmem_1, M3, Val(D1), Val(D2), Val(D), Val(nthreads), N)
    intermediate_layout_write!(C, shmem_1, Val(D1), Val(D2), Val(D), Val(nthreads), N)
end

@inline function kernel_gram!(
    Gs,
    As,
    ::Val{D1},
    ::Val{D2},
    ::Val{D},
    ::Val{nthreads},
    N::Int32,
) where {D1,D2,D,nthreads}
    n_mats_per_warp = 32i32 ÷ D
    n_warps = nthreads ÷ 32i32
    n_mats_per_block = n_warps * n_mats_per_warp
    dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

    tid = threadIdx().x
    bid = blockIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32
    warp_matrix_id = div(lid - 1i32, D) + 1i32
    d = mod1(lid, D)
    grid_mtrx_id = warp_matrix_id + (wid - 1i32) * n_mats_per_warp + (bid - 1i32) * n_mats_per_block

    warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
    shmem_elems = warp_shmem_size * n_warps
    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))

    # Load A
    intermediate_layout_load!(shmem_2, As, Val(D1), Val(D2), Val(D), Val(nthreads), N)
    interm_to_dual_transfer!(shmem_1, shmem_2, Val(D1), Val(D2), Val(D), Val(nthreads), N)

    # Create dual-access matrices
    A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
    G = DualAccessMatrix(shmem_2, Val(D), warp_matrix_id)

    if warp_matrix_id <= n_mats_per_warp && grid_mtrx_id <= N
        batch_op!(gram, G, A, d, Val(D1), Val(D2), Val(D))
    end

    # Store G
    dual_to_interm_transfer!(shmem_1, G, Val(D2), Val(D2), Val(D), Val(nthreads), N)
    intermediate_layout_write!(Gs, shmem_1, Val(D2), Val(D2), Val(D), Val(nthreads), N) 
end