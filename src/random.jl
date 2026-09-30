export BatchedRNG

"""
    BatchedRNG(seed::Integer)

Random stream for `fuse`. Pass it as an explicit argument to a scalar function
using `rand(rng, T, dims...)` or `randn(rng, T, dims...)`. Batch size comes from
other inputs, or `fuse(...; batch_size=N)` for sampling without array inputs.

Each nonempty fused launch containing sampling advances the stream once.
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
