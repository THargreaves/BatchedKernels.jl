export batch_op!

@inline function batch_op!(
    ::typeof(+),
    C::DualAccessMatrix{T},
    A::DualAccessMatrix{T},
    B::DualAccessMatrix{T},
    d::Int32,
    ::Val{D},
) where {T,D}
    for i in (1i32):D
        C[i, d] = A[i, d] + B[i, d]
    end

    return nothing
end
