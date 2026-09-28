function qr_identity_plus(C::QRTraceMatrix{T}) where {T}
    tape = _qr_trace_tape(C)
    m = size(C, 1)
    ref = emit_call!(
        tape, qr_identity_plus, NodeRef[register_wrapped!(tape, C)], TraceMatrix{T,m,m}
    )
    return TraceMatrix{T,m,m}(tape, ref)
end

function _trace_compress_residual(B, r, rest...)
    tape = _qr_trace_tape(B)
    n = size(B, 2)
    refs = NodeRef[]
    for (matrix, vector) in ((B, r), rest...)
        size(matrix, 1) == length(vector) && size(matrix, 2) == n ||
            throw(DimensionMismatch("Residual dimensions mismatch"))
        _qr_trace_tape(matrix) === tape && vector.tape === tape ||
            throw(ArgumentError("QR operands belong to different tapes"))
        push!(refs, register_wrapped!(tape, matrix), vector.ref)
    end
    T = eltype(B)
    a, b, c = emit_results!(
        tape,
        qr_compress_residual,
        refs,
        (TraceMatrix{T,n,n}, TraceVector{T,n}, TraceScalar{T}),
    )
    return TraceMatrix{T,n,n}(tape, a), TraceVector{T,n}(tape, b), TraceScalar{T}(tape, c)
end
function qr_compress_residual(B::QRTraceMatrix{T}, r::TraceVector{T}) where {T}
    return _trace_compress_residual(B, r)
end
function qr_compress_residual(
    B::QRTraceMatrix{T}, r::TraceVector{T}, C::QRTraceMatrix{T}, q::TraceVector{T}
) where {T}
    return _trace_compress_residual(B, r, (C, q))
end
