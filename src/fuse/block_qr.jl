# Narrow helper dispatch keeps CPU array assembly unchanged and records only the
# actual blocks on the tape. Logical zeros and assembled extents need no storage.
const QRTraceMatrix{T} = Union{
    TraceMatrix{T},
    Adjoint{T,<:TraceMatrix{T}},
    Transpose{T,<:TraceMatrix{T}},
    UpperTriangular{T,<:TraceMatrix{T}},
    LowerTriangular{T,<:TraceMatrix{T}},
}

function qr_upper_stack(A::QRTraceMatrix{T}, B::QRTraceMatrix{T}) where {T}
    n = size(B, 1)
    size(B, 2) == size(A, 2) == n || throw(DimensionMismatch("Invalid QR stack"))
    tape = _trace_tape(B)
    _trace_tape(A) === tape || throw(ArgumentError("QR operands belong to different tapes"))
    refs = NodeRef[register_wrapped!(tape, A), register_wrapped!(tape, B)]
    ref = emit_call!(tape, qr_upper_stack, refs, TraceMatrix{T,n,n})
    return TraceMatrix{T,n,n}(tape, ref)
end

function qr_upper_blocks(
    A::QRTraceMatrix{T}, B::QRTraceMatrix{T}, C::QRTraceMatrix{T}
) where {T}
    m, n = size(A, 1), size(C, 1)
    size(A) == (m, m) && size(B) == (n, m) && size(C) == (n, n) ||
        throw(DimensionMismatch("Invalid QR block dimensions"))
    tape = _trace_tape(A)
    _trace_tape(B) === tape && _trace_tape(C) === tape ||
        throw(ArgumentError("QR operands belong to different tapes"))
    refs = NodeRef[register_wrapped!(tape, x) for x in (A, B, C)]
    r1, r2, r3 = emit_results!(
        tape,
        qr_upper_blocks,
        refs,
        (TraceMatrix{T,m,m}, TraceMatrix{T,m,n}, TraceMatrix{T,n,n}),
    )
    return TraceMatrix{T,m,m}(tape, r1),
    TraceMatrix{T,m,n}(tape, r2),
    TraceMatrix{T,n,n}(tape, r3)
end

function covariance_root_logdet(A::QRTraceMatrix{T}) where {T}
    size(A, 1) == size(A, 2) || throw(DimensionMismatch("Root must be square"))
    tape = _trace_tape(A)
    ref = emit_call!(tape, logdet, NodeRef[register_wrapped!(tape, A)], TraceScalar{T})
    return TraceScalar{T}(tape, ref)
end
