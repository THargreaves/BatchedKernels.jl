export BatchedRNG

"""
    BatchedRNG(seed::Integer)

Random stream for `fuse` and `rand!`/`randn!` on dense `CuArray`s of Float32 or
Float64. Pass it as an explicit argument to a scalar function
using `rand(rng, T, dims...)` or `randn(rng, T, dims...)`. Batch size comes from
other inputs, or `fuse(...; batch_size=N)` for sampling without array inputs.

Each nonempty array fill or fused launch containing sampling advances the stream once.
Tracing, compilation, empty batches and unused RNG arguments do not advance it.
Use `Random.seed!(rng, seed)` to restart, or `copy(rng)` to checkpoint the stream.
Reservations are synchronized; concurrent callers get distinct counters, but
assignment of those counters to callers depends on execution order.

Samples are reproducible for the same seed and sequence of calls within a package
version, independently of launch geometry and storage policy. Splitting a batch
across calls changes the samples. CPU Julia RNG sequences and cross-version or
cross-device bitwise normal samples are not part of this contract.
"""
mutable struct BatchedRNG <: Random.AbstractRNG
    seed::UInt64
    counter::UInt64
    lock::ReentrantLock
end

function _rng_seed(seed::Integer)
    0 <= seed <= typemax(UInt64) || throw(ArgumentError("seed must fit in UInt64"))
    return UInt64(seed)
end
BatchedRNG(seed::Integer) = BatchedRNG(_rng_seed(seed), UInt64(0), ReentrantLock())

function Random.seed!(rng::BatchedRNG, seed::Integer)
    value = _rng_seed(seed)
    lock(rng.lock) do
        rng.seed = value
        return rng.counter = 0
    end
    return rng
end
function Base.copy(rng::BatchedRNG)
    return lock(rng.lock) do
        return BatchedRNG(rng.seed, rng.counter, ReentrantLock())
    end
end

# Scalar broadcast operand: batch axes are supplied by the array arguments.
Base.broadcastable(rng::BatchedRNG) = rng
Base.axes(::BatchedRNG) = ()
Base.ndims(::Type{BatchedRNG}) = 0

struct DeviceRNG
    seed::UInt64
    counter::UInt64
end

# Lock all participating objects in a stable order, including aliases only once.
# Check exhaustion before changing any stream. Reserved counters are never rolled
# back after launch submission (asynchronous device errors may arrive later).
function _reserve_rngs(rngs::Vector{BatchedRNG}, used::Vector{Bool})
    active = sort!(unique(rngs[used]); by=objectid)
    for rng in active
        lock(rng.lock)
    end
    try
        for rng in active
            rng.counter == typemax(UInt64) &&
                throw(ArgumentError("BatchedRNG counter exhausted; reseed the stream"))
        end
        states = IdDict(rng => DeviceRNG(rng.seed, rng.counter) for rng in active)
        for rng in active
            rng.counter += UInt64(1)
        end
        return [used[i] ? states[rng] : DeviceRNG(0, 0) for (i, rng) in enumerate(rngs)]
    finally
        for rng in reverse(active)
            unlock(rng.lock)
        end
    end
end

# Philox4x32-10: 64-bit seed is the key. The 128-bit counter is partitioned into
# launch (64 bits), particle (32 bits), and draw address (32 bits). Each sampling
# call owns 1024 addresses, covering the maximum 32×32 logical matrix. This
# permits 2^22 distinct sampling calls per trace without overlapping counters.
@inline function _random_words(
    rng::DeviceRNG, particle::UInt32, site::UInt32, element::UInt32
)
    key = (rng.seed % UInt32, (rng.seed >> 32) % UInt32)
    address = (site << 10) | element
    ctr = (rng.counter % UInt32, (rng.counter >> 32) % UInt32, particle, address)
    return Random123.philox(key, ctr, Val(10))
end

# Uniforms in [0,1), retaining 24 or 53 random bits respectively.
@inline _uniform(::Type{Float32}, a::UInt32, b::UInt32) = Float32(a >> 8) * Float32(0x1p-24)
@inline function _uniform(::Type{Float64}, a::UInt32, b::UInt32)
    bits = (UInt64(a) << 21) | UInt64(b >> 11)
    return Float64(bits) * 0x1p-53
end
@inline function _random_sample(
    ::Type{T},
    ::Val{Normal},
    rng::DeviceRNG,
    particle::UInt32,
    site::UInt32,
    element::UInt32,
) where {T,Normal}
    a, b, c, e = _random_words(rng, particle, site, element)
    u = _uniform(T, a, b)
    if Normal
        # 1-u is in (0,1], so log never sees zero. A fixed-size Box-Muller
        # transform uses one Philox block per sample and needs no rejection loop.
        v = _uniform(T, c, e)
        return sqrt(-T(2) * log(one(T) - u)) * cos(T(2) * T(pi) * v)
    end
    return u
end

# Ordinary array fills use the same launch reservations as fuse, but assign all
# remaining 64 counter bits to the zero-based logical linear index. Splitting
# those bits into the fused particle/site/element coordinates is injective.
@inline function _array_random_address(index::UInt64)
    return ((index >> 32) % UInt32, (index % UInt32) >> 10, UInt32(index & 0x3ff))
end

function _fill_random_kernel!(A, rng::DeviceRNG, normal::Val)
    i = UInt64(blockIdx().x - 1) * UInt64(blockDim().x) + UInt64(threadIdx().x)
    stride = UInt64(blockDim().x) * UInt64(gridDim().x)
    n = UInt64(length(A))
    while i <= n
        particle, site, element = _array_random_address(i - UInt64(1))
        @inbounds A[Int(i)] = _random_sample(eltype(A), normal, rng, particle, site, element)
        i += stride
    end
    return nothing
end

function _fill_random!(rng::BatchedRNG, A::CuArray{T}, normal::Val) where {T}
    T in (Float32, Float64) ||
        throw(ArgumentError("BatchedRNG array sampling supports Float32 and Float64"))
    # The counter mapping covers UInt64 indices; Julia arrays have the stricter
    # Int length limit. UInt64 kernel arithmetic leaves room for the final stride.
    0 <= length(A) <= typemax(Int) || throw(ArgumentError("array length must fit in Int"))
    isempty(A) && return A
    CUDA.device(A) == CUDA.device() ||
        throw(ArgumentError("the destination CuArray must be on the active CUDA device"))
    # Adaptation and compilation happen before reservation. Any subsequent launch
    # failure consumes its reservation, including asynchronously reported errors.
    kernel = @cuda launch=false _fill_random_kernel!(A, DeviceRNG(0, 0), normal)
    threads = 256
    blocks = min(cld(length(A), threads), 65535)
    state = only(_reserve_rngs([rng], [true]))
    kernel(A, state, normal; threads, blocks)
    return A
end

"""
    rand!(rng::BatchedRNG, A::CuArray)
    randn!(rng::BatchedRNG, A::CuArray)

Fill a dense Float32/Float64 GPU array with uniforms in `[0, 1)` or standard
normals. The destination must be on the active CUDA device. Array rank and shape
do not affect the logical linear random addresses; lengths up to `typemax(Int)`
are supported, subject to available memory. Noncontiguous views are not supported.

A nonempty fill consumes one launch reservation shared with `fuse`. Empty fills,
validation failures and compilation do not advance the stream. A failure after
reservation does consume it. The current task's CUDA stream is used; reseeding
does not cancel work already submitted with an immutable RNG snapshot.
"""
Random.rand!(rng::BatchedRNG, A::CuArray) = _fill_random!(rng, A, Val(false))
Random.randn!(rng::BatchedRNG, A::CuArray) = _fill_random!(rng, A, Val(true))
# Resolve the intersection with GPUArrays' AbstractRNG floating-array fallback.
Random.randn!(rng::BatchedRNG, A::CuArray{<:Union{AbstractFloat,Complex{<:AbstractFloat}}}) =
    _fill_random!(rng, A, Val(true))
