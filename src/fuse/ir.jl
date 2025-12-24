using LinearAlgebra

##########################
### IR DATA STRUCTURES ###
##########################

"SSA ID for each variable in the function."
struct ValueId
    id::Int
end

const IRArg{T} = Union{ValueId, T}

"A node in the IR program. Represents one operation."
struct IRNode{T}
    out::ValueId
    op::Symbol
    args::Vector{IRArg{T}}   # ValueId or literal (e.g. 3.0f0)
end


#######################
### SUPPORTED TYPES ###
#######################

abstract type SymKind end

# Batched types
abstract type MatKind <: SymKind end
struct DenseMatKind <: MatKind end
struct TransMatKind <: MatKind end
struct SymMatKind <: MatKind end
abstract type TrigMatKind <: MatKind end
struct LowerTrigMatKind <: TrigMatKind end
struct UpperTrigMatKind <: TrigMatKind end
struct CholeskyKind <: SymKind end
struct VecKind <: SymKind end
struct ScalarKind <: SymKind end

# Shared (constant) types
abstract type SharedKind <: SymKind end
struct SharedMatKind <: SharedKind end
struct SharedVecKind <: SharedKind end

# Union types
const MatLike = Union{MatKind, SharedMatKind}
const VecLike = Union{VecKind, SharedVecKind}

#########################
### IR PROGRAM STRUCT ###
#########################

"An IR program: inputs, nodes, outputs."
mutable struct IRProgram{T}
    inputs::Vector{Pair{Symbol,ValueId}}
    nodes::Vector{IRNode{T}}
    outputs::Vector{ValueId}
    next_id::Int
    kinds::Dict{ValueId,Type{<:SymKind}}
    D::Int
    nthreads::Int
end
IRProgram(::Type{T}, D::Int, nthreads::Int) where {T} = IRProgram{T}(
    Pair{Symbol,ValueId}[],
    IRNode{T}[],
    ValueId[],
    1,
    Dict{ValueId,Type{<:SymKind}}(),
    D,
    nthreads,
)
Base.eltype(::IRProgram{T}) where {T} = T

"Constant number type helper"
@inline function fix_type(::IRProgram{T}, x) where {T}
    x isa ValueId && return x
    x isa T && return x
    x isa Number && return T(x)
    error("Unsupported IR arg $(typeof(x))")
end

"Generate a new variable ID."
fresh!(p::IRProgram) = (vid = ValueId(p.next_id); p.next_id += 1; vid)


#######################
### SYMBOLIC VALUES ###
#######################

"""
A symbolic value used during tracing.

- `vid` identifies the value in the IR (SSA id)
- `kind` the type of the value
- `prog` the IRProgram that is recorded into
"""
abstract type AbstractSymVal end

struct SymVal{K<:SymKind,T} <: AbstractSymVal
    vid::ValueId
    # kind::K
    prog::IRProgram{T}
end
Base.getproperty(x::SymVal{K}, s::Symbol) where {K} = s === :kind ? K : getfield(x, s)
# kind(::SymVal{K}) where {K} = K

function create_symval(vid::ValueId, ::Type{K}, prog::IRProgram{T}) where {K<:SymKind,T}
    prog.kinds[vid] = K
    return SymVal{K,T}(vid, prog)
end

"""
Symbolic value for Cholesky decomposition results, containing the additional attribute `uplo`
"""
struct CholeskyVal{T} <: AbstractSymVal
    vid::ValueId
    # kind::DenseMatKind
    prog::IRProgram{T}
    uplo::Char
end

CholeskyVal(vid::ValueId, prog::IRProgram{T}) where {T} = CholeskyVal{T}(vid, prog, 'U')

function Base.getproperty(x::CholeskyVal, s::Symbol)
    s === :U && return UpperTriangular(x)
    s === :L && return LowerTriangular(x')
    s === :kind && return CholeskyKind
    return getfield(x, s)
end


########################################
### INPUT SPECIFICATIONS FOR TRACING ###
########################################

"""
Specifications for the input variables, containing information about the type and variable name.
"""
abstract type Spec end
struct MatSpec <: Spec
    name::Symbol
end
struct VecSpec <: Spec
    name::Symbol
end
struct ScalarSpec <: Spec
    name::Symbol
end
struct SharedMatSpec <: Spec
    name::Symbol
end
struct SharedVecSpec <: Spec
    name::Symbol
end

Mat(name::Symbol) = MatSpec(name)
Vec(name::Symbol) = VecSpec(name)
Scal(name::Symbol) = ScalarSpec(name)
SharedMat(name::Symbol) = SharedMatSpec(name)
SharedVec(name::Symbol) = SharedVecSpec(name)

"Map betwteen input specs and types."
spec_kind_map = Dict{Type{<:Spec}, Type{<:SymKind}}(
    MatSpec => DenseMatKind,
    SharedMatSpec => SharedMatKind,
    VecSpec => VecKind,
    SharedVecSpec => SharedVecKind,
    ScalarSpec => ScalarKind,
)

function make_input(spec::Spec, prog::IRProgram)
    vid = fresh!(prog)
    push!(prog.inputs, spec.name => vid)
    # prog.kinds[vid] = spec_kind_map[typeof(spec)]()
    return create_symval(vid, spec_kind_map[typeof(spec)], prog)
end


##########################
### IR EMISSION HELPER ###
##########################

"Append an IR node and return its output ValueId."
function emit!(prog::IRProgram{T}, op::Symbol, args::Vector) where {T}
    out = fresh!(prog)
    typed_args = IRArg{T}[fix_type(prog, arg) for arg in args]
    push!(prog.nodes, IRNode{T}(out, op, typed_args))
    return out
end


##########################
### OPERATOR OVERLOADS ###
##########################

@inline result_kind_mul(::Type{<:MatLike}, ::Type{<:MatLike}) = DenseMatKind
@inline result_kind_mul(::Type{<:MatLike}, ::Type{<:VecLike}) = VecKind
@inline result_kind_mul(::Type{<:VecLike}, ::Type{<:VecLike}) = error("Vec * Vec not yet supported")
@inline result_kind_mul(::Type{ScalarKind}, ::Type{ScalarKind}) = error("Scalar * Scalar not yet supported")
@inline result_kind_mul(::Type{ScalarKind}, ::Type{<:MatLike}) = error("Scalar * Mat not yet supported")
@inline result_kind_mul(::Type{<:MatLike}, ::Type{ScalarKind}) = error("Mat * Scalar not yet supported")
@inline result_kind_mul(::Type{ScalarKind}, ::Type{<:VecLike}) = error("Scalar * Vec not yet supported")
@inline result_kind_mul(::Type{<:VecLike}, ::Type{ScalarKind}) = error("Vec * Scalar not yet supported")
@inline result_kind_mul(::Type{<:SymKind}, ::Type{<:SymKind}) = error("Unknown combination of types to multiply")
function Base.:*(x::SymVal, y::SymVal)
    kind = result_kind_mul(x.kind, y.kind)
    out = emit!(x.prog, :mul, Any[x.vid, y.vid])
    return create_symval(out, kind, x.prog)
end

@inline result_kind_mul_const(::Type{ScalarKind}) = ScalarKind
@inline result_kind_mul_const(::Type{<:MatLike}) = DenseMatKind
@inline result_kind_mul_const(::Type{<:VecLike}) = VecKind
@inline result_kind_mul_const(::Type{<:SymKind}) = error("Unknown combination of types to multiplication by constant")
function Base.:*(x::SymVal, y::Number)
    kind = result_kind_mul_const(x.kind)
    out = emit!(x.prog, :mul_const, Any[x.vid, fix_type(x.prog, y)])
    return create_symval(out, kind, x.prog)
end
Base.:*(x::Number, y::SymVal) = y * x

function LinearAlgebra.cholesky(x::SymVal)
    x.kind <: SymMatKind || error("Unknown types for Cholesky. Ensure matrix is marked as Symmetric")
    out = emit!(x.prog, :chol, Any[x.vid])
    x.prog.kinds[out] = DenseMatKind
    return CholeskyVal(out, x.prog)
end

function LinearAlgebra.Symmetric(x::SymVal)
    x.kind <: MatLike || error("Symmetric() is only defined for matrices")
    out = emit!(x.prog, :sym, Any[x.vid])
    return create_symval(out, SymMatKind, x.prog)
end

function LinearAlgebra.LowerTriangular(x::SymVal{<:MatLike})
    out = emit!(x.prog, :lowertrig, Any[x.vid])
    return create_symval(out, LowerTrigMatKind, x.prog)
end

function LinearAlgebra.LowerTriangular(x::CholeskyVal)
    out = emit!(x.prog, :lowertrig, Any[x.vid])
    return create_symval(out, LowerTrigMatKind, x.prog)
end

function LinearAlgebra.UpperTriangular(x::SymVal{<:MatLike})
    out = emit!(x.prog, :uppertrig, Any[x.vid])
    return create_symval(out, UpperTrigMatKind, x.prog)
end

function LinearAlgebra.UpperTriangular(x::CholeskyVal)
    out = emit!(x.prog, :uppertrig, Any[x.vid])
    return create_symval(out, UpperTrigMatKind, x.prog)
end

function Base.:\(x::SymVal, y::SymVal)
    x.kind <: TrigMatKind && y.kind <: MatLike || error("Unknown combination of types to leftdiv")
    if x.kind <: LowerTrigMatKind
        out = emit!(x.prog, :forwardsolve, Any[x.vid, y.vid])
    else
        out = emit!(x.prog, :backwardsolve, Any[x.vid, y.vid])
    end
    return create_symval(out, DenseMatKind, x.prog)
end

Base.:/(x::SymVal{ScalarKind}, y::SymVal{ScalarKind}) = error("Scalar/Scalar not yet supported")
function Base.:/(x::SymVal{<:MatLike}, y::SymVal{<:SymMatKind})
    chol_res = cholesky(y)
    interm = chol_res.L \ x'
    result = chol_res.U \ interm
    return result'
end
function Base.:/(x::SymVal{<:MatLike}, y::SymVal{<:MatLike})
    error("Solves are not yet supported for non-symmetric matrices")
end
Base.:/(x::SymVal{<:SymKind}, y::SymVal{<:SymKind}) = error("Unknown combination of types to div")

function LinearAlgebra.adjoint(x::SymVal{<:MatLike})
    out = emit!(x.prog, :trans, Any[x.vid])
    return create_symval(out, TransMatKind, x.prog)
end
function LinearAlgebra.adjoint(x::CholeskyVal)
    out = emit!(x.prog, :trans, Any[x.vid])
    return create_symval(out, TransMatKind, x.prog)
end
LinearAlgebra.adjoint(x::SymVal{VecKind}) = error("Vec' not yet supported")
LinearAlgebra.adjoint(x::SymVal{ScalarKind}) = x
LinearAlgebra.adjoint(x::SymVal{<:SymKind}) = error("Unknown type to adjoint")

@inline result_kind_add_sub(::Type{<:MatLike}, ::Type{<:MatLike}) = DenseMatKind
@inline result_kind_add_sub(::Type{<:VecLike}, ::Type{<:VecLike}) = VecKind
@inline result_kind_add_sub(::Type{ScalarKind}, ::Type{ScalarKind}) = error("Scalar+-Scalar not yet supported")
@inline result_kind_add_sub(::Type{<:SymKind}, ::Type{<:SymKind}) = error("Unknown combination of types to add/sub")
function Base.:+(x::SymVal, y::SymVal)
    kind = result_kind_add_sub(x.kind, y.kind)
    out = emit!(x.prog, :add, Any[x.vid, y.vid])
    return create_symval(out, kind, x.prog)
end

function Base.:-(x::SymVal, y::SymVal)
    kind = result_kind_add_sub(x.kind, y.kind)
    out = emit!(x.prog, :sub, Any[x.vid, y.vid])
    return create_symval(out, kind, x.prog)
end

function Base.:+(x::UniformScaling, y::SymVal)
    y.kind <: MatLike || error("Identity +- variable only supported for matrices")
    # x.λ == 1.0f0 || error("Non-one coefficients not yet supported")    
    out = emit!(y.prog, :iplus, Any[y.vid, fix_type(y.prog, x.λ)])
    return create_symval(out, DenseMatKind, y.prog)
end

Base.:+(x::SymVal, y::UniformScaling) = y + x

function Base.:-(x::UniformScaling, y::SymVal)
    y.kind <: MatLike || error("Identity +- variable only supported for matrices")
    # x.λ == 1.0f0 || error("Non-one coefficients not yet supported")    
    out = emit!(y.prog, :iminus, Any[y.vid, fix_type(y.prog, x.λ)])
    return create_symval(out, DenseMatKind, y.prog)
end

Base.:-(x::SymVal, y::UniformScaling) = error("Val - identity not yet supported")  # -y + x

Base.:+(x::SymVal) = x

Base.:-(x::SymVal) = fix_type(x.prog, -1) * x

"Normalise return value to Vector{SymVal}."
function _as_symvals(out)
    if out isa AbstractSymVal
        return AbstractSymVal[out]
    elseif out isa Tuple
        all(x -> x isa AbstractSymVal, out) || error("All outputs must be SymVal")
        return collect(out)
    elseif out === nothing
        return AbstractSymVal[]
    else
        error("Return must be SymVal or Tuple of SymVals")
    end
end


###########################
### TRACING ENTRY POINT ###
###########################

"""
Trace a function `f` on symbolic inputs specified by `specs`.

Example:
    prog = trace(func, (Mat(:A), Mat(:B), Mat(:C), Scal(:a), Scal(:b)))
"""
function trace(f, specs::Tuple; T::Type, D::Int, nthreads::Int)
    prog = IRProgram(T, D, nthreads)

    syms = map(s -> make_input(s, prog), specs)

    out = f(syms...)
    outs = _as_symvals(out)

    prog.outputs = [o.vid for o in outs]
    return prog
end


#######################
### PRETTY PRINTING ###
#######################

function _fmt_arg(inputs_map::Dict{ValueId,Symbol}, a)
    if a isa ValueId
        if haskey(inputs_map, a)
            return String(inputs_map[a])
        else
            return "%" * string(a.id)
        end
    else
        return repr(a)
    end
end

function Base.show(io::IO, prog::IRProgram)
    # Map ValueId of inputs -> their names
    inputs_map = Dict{ValueId,Symbol}()
    for (name, vid) in prog.inputs
        inputs_map[vid] = name
    end

    println(io, "IRProgram(")
    println(io, "  inputs:")
    for (name, vid) in prog.inputs
        println(io, "    ", name, " => %", vid.id)
    end
    println(io, "  nodes:")
    for node in prog.nodes
        args_s = join((_fmt_arg(inputs_map, arg) for arg in node.args), ", ")
        println(io, "    %", node.out.id, " = ", node.op, "(", args_s, ")")
    end
    outs_s = join("%" .* string.(getfield.(prog.outputs, :id)), ", ")
    println(io, "  outputs: [", outs_s, "]")
    println(io, "  D: $(prog.D)")
    println(io, "  nthreads: $(prog.nthreads)")
    print(io, ")")
end