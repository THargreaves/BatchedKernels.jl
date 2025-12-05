@inline function _get_large_n_mats_per_block(
    ::Val{D}, ::Val{nthreads}, ::Val{:conseq}
) where {D,nthreads}
    return nthreads ÷ (D * D)
end

@inline function _get_large_n_mats_per_block(
    ::Val{D}, ::Val{nthreads}, ::Val{:indep}
) where {D,nthreads}
    return 1i32
end

@inline function _load_mat(::Val{false}, M)
    return M
end

@inline function _load_mat(::Val{true}, M)
    return M'
end

@inline function _check_large_boundary(
    block_mtrx_id::Int32,
    n_mats_per_block::Int32,
    grid_mtrx_id::Int32,
    N::Int32,
    ::Val{:conseq},
    ::Val{D},
) where {D}
    return block_mtrx_id <= n_mats_per_block && grid_mtrx_id <= N
end

@inline function _check_large_boundary(
    block_mtrx_id::Int32,
    n_mats_per_block::Int32,
    grid_mtrx_id::Int32,
    N::Int32,
    ::Val{:indep},
    ::Val{D},
) where {D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)

    n_cols_per_warp = max(1i32, prevpow(2i32, 32i32 ÷ D))
    lanes_per_slot = 32i32 ÷ n_cols_per_warp
    lane_slot_id = ((lid - 1i32) % lanes_per_slot) + 1i32

    return lane_slot_id <= D && block_mtrx_id <= n_mats_per_block && grid_mtrx_id <= N
end

@inline function _calc_large_mtrx_id(::Val{D}, ::Val{:conseq}) where {D}
    tid = threadIdx().x
    return div(tid - 1i32, D * D) + 1i32
end

@inline function _calc_large_mtrx_id(::Val{D}, ::Val{:indep}) where {D}
    tid = threadIdx().x
    lid = mod1(tid, 32i32)
    wid = div(tid - 1i32, 32i32) + 1i32

    n_cols_per_warp = max(1i32, prevpow(2i32, 32i32 ÷ D))
    lanes_per_slot = 32i32 ÷ n_cols_per_warp
    global_col_no = (wid - 1i32) * n_cols_per_warp + div(lid - 1i32, lanes_per_slot) + 1i32

    return div(global_col_no - 1i32, D) + 1i32
end

"""
Matrix multiplication kernel for the case where D^2 threads handle one matrix
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
    ::Val{:large},
    ::Val{mode},
) where {D,nthreads,A_adj,B_adj,mode}
    bid = blockIdx().x

    n_mats_per_block = _get_large_n_mats_per_block(Val(D), Val(nthreads), Val(mode))
    interm_pad_freq = div(32i32, D & -D) * D
    block_mtrx_id = _calc_large_mtrx_id(Val(D), Val(mode))
    grid_mtrx_id = (bid - 1i32) * n_mats_per_block + block_mtrx_id

    padded_amount_per_block = (n_mats_per_block * D * D - 1i32) ÷ interm_pad_freq
    shmem_elems = D * D * n_mats_per_block + padded_amount_per_block

    shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
    shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

    intermediate_layout_load!(shmem_1, As, Val(D), Val(nthreads), N, Val(:large), Val(mode))
    intermediate_layout_load!(shmem_2, Bs, Val(D), Val(nthreads), N, Val(:large), Val(mode))

    sync_threads()
    if _check_large_boundary(
        block_mtrx_id, n_mats_per_block, grid_mtrx_id, N, Val(mode), Val(D)
    )
        A = DualAccessMatrix(shmem_1, Val(D), block_mtrx_id, Val(:large))
        B = DualAccessMatrix(shmem_2, Val(D), block_mtrx_id, Val(:large))
        C = DualAccessMatrix(shmem_3, Val(D), block_mtrx_id, Val(:large))

        A_mat = _load_mat(Val(A_adj), A)
        B_mat = _load_mat(Val(B_adj), B)
        batch_op!(*, C, A_mat, B_mat, Val(D), Val(:large), Val(mode))
    end
    sync_threads()

    intermediate_layout_write!(
        Cs, shmem_3, Val(D), Val(nthreads), N, Val(:large), Val(mode)
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