# Host-side contracts for the new compute bodies. This registry is deliberately
# separate from legacy emission until assignment validation is integrated (M5).
# An empty result means no hybrid body is supported; it is NOT evidence that a
# legacy body supports the requested shape or wrapper.

const _VARIANT_INPUT_RESIDENCES = (:single, :dual, :register, :shared_input)
const _VARIANT_OUTPUT_RESIDENCES = (:single, :dual, :register)

struct OrientationVariant{I<:Tuple,R<:Tuple,A<:Tuple}
    id::Symbol
    input_access::I
    output_access::Union{Symbol,Tuple}
    input_residences::R
    output_residences::Tuple{Symbol,Symbol,Symbol}
    body::Symbol
    mirror::Symbol
    shape_rule::Symbol
    wrapper_rule::Symbol
    alias_safe_args::A
    alias_requirement::Symbol
    alias_residences::Tuple{Symbol,Symbol}
    synchronization::Symbol
    participation::Symbol
end

function _orientation_variant(id, inputs, output, shape_rule; mirror=:identity, aliases=())
    return OrientationVariant(
        id,
        inputs,
        output,
        map(_ -> _VARIANT_INPUT_RESIDENCES, inputs),
        _VARIANT_OUTPUT_RESIDENCES,
        id,
        mirror,
        shape_rule,
        :logical_inputs_dense_output,
        aliases,
        :all_overlaps_pointwise_identical,
        (:single, :dual),
        :caller_entry_and_exit_shared_fences,
        :complete_matrix_group,
    )
end

# Only these logical wrappers have audited explicit accessors. Trace-only IAddSub
# carries its shape in its type; its concrete getter parent is supplied by codegen.
_variant_shape(::Type) = nothing
_variant_shape(::Type{TraceMatrix{T,M,N}}) where {T,M,N} = (M, N)
_variant_shape(::Type{SingleAccessMatrix{T,M,N,D,O,S}}) where {T,M,N,D,O,S} = (M, N)
_variant_shape(::Type{RegisterMatrix{T,M,N,D,O,L}}) where {T,M,N,D,O,L} = (M, N)
_variant_shape(::Type{DualAccessMatrix{T,D}}) where {T,D} = (D, D)
_variant_shape(::Type{SharedMatrix{T,M,N,P}}) where {T,M,N,P} = (M, N)
_variant_shape(::Type{IAddSubWrapped{T,D}}) where {T,D} = (D, D)
_variant_shape(::Type{<:IAddSubGetterMatrix{T,D,P}}) where {T,D,P} = _variant_shape(P)
function _variant_shape(::Type{<:Union{Adjoint{T,P},Transpose{T,P}}}) where {T,P}
    s = _variant_shape(P)
    return s === nothing ? nothing : reverse(s)
end
function _variant_shape(
    ::Type{
        <:Union{
            LowerTriangular{T,P},
            UpperTriangular{T,P},
            UnitLowerTriangular{T,P},
            UnitUpperTriangular{T,P},
        },
    },
) where {T,P}
    return _variant_shape(P)
end

_variant_eltype(::Type{IAddSubWrapped{T,D}}) where {T,D} = T
_variant_eltype(T::Type) = eltype(T)
function _variant_input_domain(types; element_types=(Float32, Float64))
    shapes = map(_variant_shape, types)
    return all(s -> s !== nothing && all(d -> 1 <= d <= 32, s), shapes) &&
           all(T -> _variant_eltype(T) in element_types, types) &&
           all(T -> _variant_eltype(T) === _variant_eltype(first(types)), types)
end

_variant_triangular(::Type) = false
function _variant_triangular(
    ::Type{<:Union{LowerTriangular,UpperTriangular,UnitLowerTriangular,UnitUpperTriangular}}
)
    return true
end
function _variant_triangular(::Type{<:Union{Adjoint{T,P},Transpose{T,P}}}) where {T,P}
    return _variant_triangular(P)
end

"""
    orientation_variants(fn, argtypes...)

Return deterministic contracts for Float32 and Float64 hybrid matrix bodies.
Operands must have matching element types. `:row`
means RowAccess, `:col` ColAccess, and `:any` means broadcast-only input (either
orientation, still subject to shape/mask rules). The result describes fresh dense
outputs; multi-result variants specify an output-access tuple in result order. Aliases are optional only at the listed operand positions in single/dual
storage, with identical unwrapped shape/map; registers never alias tape operands.
Candidate positions alone never authorize reuse: every other input overlapping the
output must read the same pointwise element-to-lane/address map. In particular,
reusing A for A + A' is unsafe. M5 must validate this across all wrapper/owner aliases.

All lanes of each complete D_MAX-wide matrix group call the body, including lanes
without an output line. Callers fence shared producers before and shared consumers
or slot reuse after the call. Current factorization bodies use private scratch and
introduce no internal shared-memory dependency. The row solve's scalar-use compiler
constraint is not a shared-memory fence. The column solve is a separate algorithm:
factor/RHS/output all require ColAccess, and all lanes broadcast solved RHS pivots.

An empty tuple retains the existing legacy path and its existing domain checks.
This registry does not expand legacy support for unknown operations, wrappers,
element types, or shapes. Forced-in-place operations remain on that shared path.
"""
orientation_variants(::Any, ::Type...) = ()

function orientation_variants(::typeof(*), A::Type, B::Type)
    if B <: TraceVector
        _variant_input_domain((A,)) || return ()
        eltype(A) === eltype(B) && _variant_shape(A)[2] == shape(B)[1] || return ()
        return (_orientation_variant(:matvec_col, (:col,), :none, :matvec),)
    end
    _variant_input_domain((A, B); element_types=(Float32, Float64)) || return ()
    _variant_eltype(A) === _variant_eltype(B) || return ()
    _variant_shape(A)[2] == _variant_shape(B)[1] || return ()
    return (
        _orientation_variant(:matmul_row, (:any, :row), :row, :matmul),
        _orientation_variant(
            :matmul_col, (:col, :any), :col, :matmul; mirror=:adjoint_swap
        ),
    )
end
function orientation_variants(fn::Union{typeof(+),typeof(-)}, A::Type, B::Type)
    _variant_input_domain((A, B)) || return ()
    _variant_shape(A) == _variant_shape(B) || return ()
    # Reusing a wrapped input's storage requires a separate semantic alias audit.
    aliases = Tuple(
        i for (i, T) in enumerate((A, B)) if
        T <: Union{TraceMatrix,SingleAccessMatrix,DualAccessMatrix}
    )
    prefix = fn === (+) ? :add : :sub
    return (
        _orientation_variant(
            Symbol(prefix, :_row), (:row, :row), :row, :same_shape; aliases
        ),
        _orientation_variant(
            Symbol(prefix, :_col), (:col, :col), :col, :same_shape; mirror=:adjoint, aliases
        ),
    )
end
function orientation_variants(::typeof(/), A::Type, B::Type)
    _variant_input_domain((A,)) || return ()
    (B <: Number && (B <: TraceScalar ? eltype(B) : B) === eltype(A)) || return ()
    return (
        _orientation_variant(:divide_row, (:row,), :row, :matrix_divide),
        _orientation_variant(:divide_col, (:col,), :col, :matrix_divide),
    )
end

function orientation_variants(::typeof(one), A::Type)
    _variant_input_domain((A,)) || return ()
    m, n = _variant_shape(A)
    m == n || return ()
    return (
        _orientation_variant(:identity_row, (:any,), :row, :matrix_identity),
        _orientation_variant(:identity_col, (:any,), :col, :matrix_identity),
    )
end

function orientation_variants(::typeof(cholesky), A::Type)
    _variant_input_domain((A,)) || return ()
    s = _variant_shape(A)
    s[1] == s[2] || return ()
    return (_orientation_variant(:cholesky_row, (:row,), :row, :square_spd),)
end
function orientation_variants(::typeof(\), A::Type, B::Type)
    if B <: TraceVector
        _variant_input_domain((A,)) && _variant_triangular(A) || return ()
        a = _variant_shape(A)
        eltype(A) === eltype(B) && a[1] == a[2] == shape(B)[1] || return ()
        return (_orientation_variant(:solve_vector_col, (:col,), :none, :vector_solve),)
    end
    _variant_input_domain((A, B)) && _variant_triangular(A) || return ()
    a, b = _variant_shape(A), _variant_shape(B)
    a[1] == a[2] == b[1] || return ()
    return (
        _orientation_variant(:solve_row, (:any, :row), :row, :triangular_solve),
        _orientation_variant(:solve_col, (:col, :col), :col, :triangular_solve),
    )
end

function orientation_variants(::typeof(symmetric_part), A::Type)
    _variant_input_domain((A,)) || return ()
    m, n = _variant_shape(A)
    m == n || return ()
    return (_orientation_variant(:symmetric_row, (:both,), :row, :square_symmetric),)
end

function orientation_variants(::typeof(logdet), A::Type)
    _variant_input_domain((A,)) || return ()
    m, n = _variant_shape(A)
    m == n || return ()
    return (_orientation_variant(:logdet_any, (:any,), :none, :scalar_logdet),)
end

"""Emit a validated variant call; assignment/residence and alias validation belongs to M5."""
function emit_variant(v::OrientationVariant, dest, args::Vector, types::Vector, D_MAX::Int)
    1 <= D_MAX <= 32 || throw(ArgumentError("D_MAX must be in 1:32"))
    length(args) == length(types) || throw(ArgumentError("variant arity mismatch"))
    matrix_types = filter(t -> _variant_shape(t) !== nothing, types)
    length(matrix_types) == length(v.input_access) ||
        throw(ArgumentError("variant matrix operand mismatch"))
    shapes = map(_variant_shape, matrix_types)
    all(s -> maximum(s) <= D_MAX, shapes) ||
        throw(ArgumentError("variant shape exceeds D_MAX"))
    fn = if v.shape_rule in (:matmul, :matvec)
        (*)
    elseif v.shape_rule === :matrix_divide
        (/)
    elseif v.shape_rule === :matrix_identity
        one
    elseif v.shape_rule === :square_spd
        cholesky
    elseif v.shape_rule in (:triangular_solve, :vector_solve)
        (\)
    elseif v.shape_rule === :square_symmetric
        symmetric_part
    elseif v.shape_rule === :qr_stack
        qr_upper_stack
    elseif v.shape_rule === :qr_blocks
        qr_upper_blocks
    elseif v.shape_rule === :qr_identity
        qr_identity_plus
    elseif v.shape_rule === :qr_residual
        qr_compress_residual
    elseif v.shape_rule === :scalar_logdet
        logdet
    elseif startswith(String(v.id), "add")
        (+)
    else
        (-)
    end
    any(candidate -> candidate == v, orientation_variants(fn, types...)) ||
        throw(ArgumentError("unsupported variant/type combination"))
    if v.shape_rule === :scalar_logdet
        n = shapes[1][1]
        return :($dest = variant_logdet($(args[1]), d, Val(Int32($n)), Val(Int32($D_MAX))))
    end
    if v.shape_rule === :qr_residual
        m, n = shapes[1]
        p = length(shapes) == 2 ? shapes[2][1] : 0
        call = Expr(
            :call,
            :variant_compress_residual!,
            dest.args[1],
            dest.args[2],
            args...,
            :d,
            [:(Val(Int32($x))) for x in (m, n, p, D_MAX)]...,
        )
        return dest.args[3] === nothing ? call : :($(dest.args[3]) = $call)
    end
    dims = if v.shape_rule === :matmul
        (shapes[1][1], shapes[1][2], shapes[2][2], D_MAX)
    elseif v.shape_rule === :qr_stack
        (shapes[1][1], shapes[2][1], D_MAX)
    elseif v.shape_rule === :qr_blocks
        (shapes[1][1], shapes[3][1], D_MAX)
    elseif v.shape_rule in (:square_spd, :square_symmetric, :vector_solve)
        (shapes[1][1], D_MAX)
    elseif v.shape_rule === :triangular_solve
        (shapes[2]..., D_MAX)
    else
        (shapes[1]..., D_MAX)
    end
    return Expr(
        :call,
        :variant_op!,
        :(Val($(QuoteNode(v.body)))),
        dest,
        args...,
        :d,
        [:(Val(Int32($n))) for n in dims]...,
    )
end

function orientation_variants(::typeof(qr_upper_stack), A::Type, B::Type)
    _variant_input_domain((A, B)) || return ()
    r, n = _variant_shape(A)
    _variant_shape(B) == (n, n) || return ()
    return (
        _orientation_variant(:qr_stack_row, (:row, :row), :row, :qr_stack),
        _orientation_variant(:qr_stack_col, (:col, :col), :col, :qr_stack),
    )
end

function orientation_variants(::typeof(qr_upper_blocks), A::Type, B::Type, C::Type)
    _variant_input_domain((A, B, C)) || return ()
    m, n = _variant_shape(B)[2], _variant_shape(B)[1]
    _variant_shape(A) == (m, m) && _variant_shape(C) == (n, n) || return ()
    return (
        _orientation_variant(
            :qr_blocks_row, (:row, :row, :row), (:row, :row, :row), :qr_blocks
        ),
        _orientation_variant(
            :qr_blocks_col, (:col, :col, :col), (:col, :col, :col), :qr_blocks
        ),
    )
end

function orientation_variants(::typeof(qr_identity_plus), A::Type)
    _variant_input_domain((A,)) || return ()
    return (_orientation_variant(:qr_identity_col, (:col,), :row, :qr_identity),)
end

function orientation_variants(
    ::typeof(qr_compress_residual), A::Type, a::Type, rest::Type...
)
    length(rest) in (0, 2) || return ()
    mats = isempty(rest) ? (A,) : (A, rest[1])
    vecs = isempty(rest) ? (a,) : (a, rest[2])
    _variant_input_domain(mats) || return ()
    n = _variant_shape(A)[2]
    all(
        _variant_shape(m)[2] == n &&
            v <: TraceVector &&
            shape(v) == (_variant_shape(m)[1],) &&
            eltype(v) === eltype(A) for (m, v) in zip(mats, vecs)
    ) || return ()
    return (
        _orientation_variant(
            :qr_residual_row, map(_ -> :row, mats), (:row, :none, :none), :qr_residual
        ),
    )
end
