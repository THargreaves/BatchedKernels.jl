@testitem "Vector Addition" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_vec_add!(
        zs, xs, ys, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_vecs_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_vector_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = D * n_warps * n_vecs_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load x and y
        vector_load!(shmem_1, xs, Val(D), Val(nthreads), N)
        vector_load!(shmem_2, ys, Val(D), Val(nthreads), N)

        # Create batched vectors
        x = BatchedVector(shmem_1, Val(D), warp_vector_id)
        y = BatchedVector(shmem_2, Val(D), warp_vector_id)
        z = BatchedVector(shmem_3, Val(D), warp_vector_id)

        if warp_vector_id <= n_vecs_per_warp
            # Perform operation
            batch_op!(+, z, x, y, d, Val(D))
        end

        # Store z
        vector_write!(zs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in SubKernelShapes.square(16)
        nblocks = cld(N, nthreads//32 * (32 ÷ D))
        CUDA.seed!(1234)

        xs = CUDA.rand(Float32, D, N)
        ys = CUDA.rand(Float32, D, N)
        zs = CUDA.zeros(Float32, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_vec_add!(
            zs, xs, ys, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        xs_cpu = Array(xs)
        ys_cpu = Array(ys)
        zs_cpu = xs_cpu .+ ys_cpu

        max_error = maximum(abs.(Array(zs) .- zs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Vector Subtraction" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_vec_sub!(
        zs, xs, ys, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_vecs_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_vector_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = D * n_warps * n_vecs_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load x and y
        vector_load!(shmem_1, xs, Val(D), Val(nthreads), N)
        vector_load!(shmem_2, ys, Val(D), Val(nthreads), N)

        # Create batched vectors
        x = BatchedVector(shmem_1, Val(D), warp_vector_id)
        y = BatchedVector(shmem_2, Val(D), warp_vector_id)
        z = BatchedVector(shmem_3, Val(D), warp_vector_id)

        if warp_vector_id <= n_vecs_per_warp
            # Perform operation
            batch_op!(-, z, x, y, d, Val(D))
        end

        # Store z
        vector_write!(zs, shmem_3, Val(D), Val(nthreads), N)

        return nothing
    end

    for D in SubKernelShapes.square(16)
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        CUDA.seed!(1234)

        xs = CUDA.rand(Float32, D, N)
        ys = CUDA.rand(Float32, D, N)
        zs = CUDA.zeros(Float32, D, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_vec_sub!(
            zs, xs, ys, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        xs_cpu = Array(xs)
        ys_cpu = Array(ys)
        zs_cpu = xs_cpu .- ys_cpu

        max_error = maximum(abs.(Array(zs) .- zs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Matrix-Vector Multiplication" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_matvec!(
        ys, As, xs, A_adj::Bool, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D,nthreads}
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_mat_elems = warp_shmem_size * n_warps
        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
        shmem_4 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        # Load A
        intermediate_layout_load!(shmem_2, As, Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(shmem_1, shmem_2, Val(D), Val(nthreads), N)

        # Load x
        vector_load!(shmem_3, xs, Val(D), Val(nthreads), N)

        # Create dual-access matrix and batched vectors
        if warp_matrix_id <= n_mats_per_warp
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
            x = BatchedVector(shmem_3, Val(D), warp_matrix_id)
            y = BatchedVector(shmem_4, Val(D), warp_matrix_id)

            # Perform operation: y = A * x or y = A' * x
            if A_adj
                batch_op!(*, y, A', x, d, Val(D))
            else
                batch_op!(*, y, A, x, d, Val(D))
            end
        end

        # Store y
        vector_write!(ys, shmem_4, Val(D), Val(nthreads), N)

        return nothing
    end

    # Test both A * x and A' * x
    test_cases = [false, true]

    for D in SubKernelShapes.square(16)
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D, D, N)
        xs = CUDA.rand(Float32, D, N)
        As_cpu = Array(As)
        xs_cpu = Array(xs)

        for A_adj in test_cases
            ys = CUDA.zeros(Float32, D, N)

            CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matvec!(
                ys, As, xs, A_adj, Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
            )

            # CPU comparison
            ys_cpu = similar(xs_cpu)
            for i in 1:N
                if A_adj
                    ys_cpu[:, i] = As_cpu[:, :, i]' * xs_cpu[:, i]
                else
                    ys_cpu[:, i] = As_cpu[:, :, i] * xs_cpu[:, i]
                end
            end

            max_error = maximum(abs.(Array(ys) .- ys_cpu))
            @test max_error < 1e-5
        end
    end
end

@testitem "Vector Addition (non-square)" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_vec_add!(
        zs, xs, ys, ::Val{D1}, ::Val{D}, ::Val{nthreads}, N::Int32
    ) where {D1,D,nthreads}
        n_vecs_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_vector_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        shmem_elems = D * n_warps * n_vecs_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_elems,))

        # Load x and y
        vector_load!(shmem_1, xs, Val(D1), Val(D), Val(nthreads), N)
        vector_load!(shmem_2, ys, Val(D1), Val(D), Val(nthreads), N)

        # Create batched vectors
        x = BatchedVector(shmem_1, Val(D), warp_vector_id)
        y = BatchedVector(shmem_2, Val(D), warp_vector_id)
        z = BatchedVector(shmem_3, Val(D), warp_vector_id)

        if warp_vector_id <= n_vecs_per_warp
            # Perform operation
            batch_op!(+, z, x, y, d, Val(D1), Val(D), Val(D))
        end

        # Store z
        vector_write!(zs, shmem_3, Val(D1), Val(D), Val(nthreads), N)

        return nothing
    end

    for (D1, D) in SubKernelShapes.padded(16)
        nblocks = cld(N, nthreads//32 * (32 ÷ D))
        CUDA.seed!(1234)

        xs = CUDA.rand(Float32, D1, N)
        ys = CUDA.rand(Float32, D1, N)
        zs = CUDA.zeros(Float32, D1, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_vec_add!(
            zs, xs, ys, Val(Int32(D1)), Val(Int32(D)), Val(Int32(nthreads)), Int32(N)
        )

        # CPU comparison
        xs_cpu = Array(xs)
        ys_cpu = Array(ys)
        zs_cpu = xs_cpu .+ ys_cpu

        max_error = maximum(abs.(Array(zs) .- zs_cpu))
        @test max_error < 1e-5
    end
end

@testitem "Matrix-Vector Multiplication (non-square)" setup = [SubKernelShapes] begin
    using BatchedKernels
    using CUDA
    using CUDA: i32
    using LinearAlgebra

    # Test parameters
    nthreads = 2^8
    N = 2^12 + 113

    @inline function kernel_matvec!(
        ys, As, xs, ::Val{D1}, ::Val{D2}, ::Val{nthreads}, N::Int32
    ) where {D1,D2,nthreads}
        D = max(D1, D2)
        n_mats_per_warp = 32i32 ÷ D
        n_warps = nthreads ÷ 32i32
        dual_padding = mod(n_mats_per_warp - mod(n_mats_per_warp * D, 32i32), 32i32)

        tid = threadIdx().x
        lid = mod1(tid, 32i32)
        warp_matrix_id = div(lid - 1i32, D) + 1i32
        d = mod1(lid, D)

        warp_shmem_size = n_mats_per_warp * D * D + dual_padding * (D - 1i32)
        shmem_mat_elems = warp_shmem_size * n_warps
        shmem_vec_elems = D * n_warps * n_mats_per_warp
        shmem_1 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_2 = CuStaticSharedArray(Float32, (shmem_mat_elems,))
        shmem_3 = CuStaticSharedArray(Float32, (shmem_vec_elems,))
        shmem_4 = CuStaticSharedArray(Float32, (shmem_vec_elems,))

        # Load A
        intermediate_layout_load!(shmem_2, As, Val(D1), Val(D2), Val(D), Val(nthreads), N)
        interm_to_dual_transfer!(
            shmem_1, shmem_2, Val(D1), Val(D2), Val(D), Val(nthreads), N
        )

        # Load x
        vector_load!(shmem_3, xs, Val(D2), Val(D), Val(nthreads), N)

        # Create dual-access matrix and batched vectors
        if warp_matrix_id <= n_mats_per_warp
            A = DualAccessMatrix(shmem_1, Val(D), warp_matrix_id)
            x = BatchedVector(shmem_3, Val(D), warp_matrix_id)
            y = BatchedVector(shmem_4, Val(D), warp_matrix_id)

            batch_op!(*, y, A, x, d, Val(D1), Val(D2), Val(D))
        end

        # Store y
        vector_write!(ys, shmem_4, Val(D1), Val(D), Val(nthreads), N)

        return nothing
    end

    for (D1, D2) in SubKernelShapes.rectangular(16)
        D = max(D1, D2)
        nblocks = cld(N, nthreads//32 * (32 ÷ D))

        CUDA.seed!(1234)

        As = CUDA.rand(Float32, D1, D2, N)
        xs = CUDA.rand(Float32, D2, N)
        As_cpu = Array(As)
        xs_cpu = Array(xs)

        ys = CUDA.zeros(Float32, D1, N)

        CUDA.@sync @cuda threads = nthreads blocks = nblocks kernel_matvec!(
            ys, As, xs, Val(Int32(D1)), Val(Int32(D2)), Val(Int32(nthreads)), Int32(N)
        )
        ys_result = Array(ys)

        # CPU comparison
        ys_cpu = similar(ys_result)
        for i in 1:N
            ys_cpu[:, i] = As_cpu[:, :, i] * xs_cpu[:, i]
        end

        max_error = maximum(abs.(ys_result .- ys_cpu))
        @test max_error < 1e-5
    end
end
