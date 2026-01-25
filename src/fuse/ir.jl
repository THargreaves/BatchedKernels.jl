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


##################
### SHAPE TYPE ###
##################

abstract type Shape end
struct MatShape <: Shape
    D1::Int
    D2::Int
end
struct VecShape <: Shape
    D1::Int
end
struct ScalShape <: Shape end
struct ConstShape <: Shape end


#######################
### SUPPORTED TYPES ###
#######################

abstract type SymKind end

# Batched types
abstract type BatchedKind <: SymKind end
abstract type MatKind <: BatchedKind end
struct DenseMatKind <: MatKind end
struct TransMatKind <: MatKind end
struct SymMatKind <: MatKind end
abstract type TrigMatKind <: MatKind end
struct LowerTrigMatKind <: TrigMatKind end
struct UpperTrigMatKind <: TrigMatKind end
struct CholeskyKind <: BatchedKind end
struct QRKind <: BatchedKind end
struct VecKind <: BatchedKind end
struct ScalarKind <: BatchedKind end
struct ConstKind <: BatchedKind end

# Shared (constant) types
abstract type SharedKind <: SymKind end
struct SharedMatKind <: SharedKind end
struct SharedVecKind <: SharedKind end
struct SharedScalarKind <: SharedKind end

# Union types
const MatLike = Union{MatKind,SharedMatKind}
const VecLike = Union{VecKind,SharedVecKind}
const ScalLike = Union{ScalarKind,SharedScalarKind}

#########################
### IR PROGRAM STRUCT ###
#########################

mutable struct IRProgram{T}
    inputs::Vector{Pair{Symbol,ValueId}}
    nodes::Vector{IRNode{T}}
    outputs::Vector{ValueId}
    next_id::Int
    kinds::Dict{ValueId,Type{<:SymKind}}
    shapes::Dict{ValueId,Shape}
    D::Int
    nthreads::Int
end
IRProgram(::Type{T}, D::Int, nthreads::Int) where {T} = IRProgram{T}(
    Pair{Symbol,ValueId}[],
    IRNode{T}[],
    ValueId[],
    1,
    Dict{ValueId,Type{<:SymKind}}(),
    Dict{ValueId,Shape}(),
    D,
    nthreads,
)
Base.eltype(::IRProgram{T}) where {T} = T

"Constant number type helper."
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
- `prog` the IRProgram that is recorded into
"""
abstract type AbstractSymVal end

struct SymVal{K<:SymKind,T} <: AbstractSymVal
    vid::ValueId
    prog::IRProgram{T}
end
Base.getproperty(x::SymVal{K}, s::Symbol) where {K} = s === :kind ? K : getfield(x, s)

function create_symval(vid::ValueId, ::Type{K}, shape::Shape, prog::IRProgram{T}) where {K<:SymKind,T}
    prog.kinds[vid] = K
    prog.shapes[vid] = shape
    return SymVal{K,T}(vid, prog)
end

abstract type FactorisationVal <: AbstractSymVal end

"""
Symbolic value for Cholesky decomposition results.
"""
struct CholeskyVal{T} <: FactorisationVal
    content::SymVal
    prog::IRProgram{T}
end

function create_choleskyval(vid::ValueId, shape::MatShape, prog::IRProgram{T}) where {T}
    # Creating SymVal for the result of Cholesky decomposition (upper triangular)
    content = create_symval(vid, DenseMatKind, shape, prog)

    return CholeskyVal{T}(content, prog)
end

function Base.getproperty(x::CholeskyVal, s::Symbol)
    s === :U && return UpperTriangular(x.content)
    s === :L && return LowerTriangular(x.content')
    s === :kind && return CholeskyKind
    s === :vid && error("Cannot access vid of a FactorisationVal")
    return getfield(x, s)
end

"""
Symbolic value for QR decomposition results, containing Q and R
"""
struct QRVal{T} <: AbstractSymVal
    Q::SymVal{<:MatKind,T}
    R::SymVal{UpperTrigMatKind,T}
    prog::IRProgram{T}
end

function create_qrval(Q::SymVal{<:MatKind,T}, R::SymVal{UpperTrigMatKind,T}, prog::IRProgram{T}) where {T}    
    # No need to create SymVals for Q and R as they already exist, just link them to this QRVal instance
    return QRVal{T}(Q, R, prog)
end

function Base.getproperty(x::QRVal, s::Symbol)
    # s === :Q && return x.Q
    s === :R && return UpperTriangular(getfield(x, :R))
    s === :kind && return QRKind
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
    D1::Int
    D2::Int
end
struct VecSpec <: Spec
    name::Symbol
    D1::Int
end
struct ScalarSpec <: Spec
    name::Symbol
end
struct SharedScalarSpec <: Spec
    name::Symbol
end
struct SharedMatSpec <: Spec
    name::Symbol
    D1::Int
    D2::Int
end
struct SharedVecSpec <: Spec
    name::Symbol
    D1::Int
end

Mat(name::Symbol, D1::Int, D2::Int) = MatSpec(name, D1, D2)
Vec(name::Symbol, D1::Int) = VecSpec(name, D1)
Scal(name::Symbol) = ScalarSpec(name)
SharedScal(name::Symbol) = SharedScalarSpec(name)
SharedMat(name::Symbol, D1::Int, D2::Int) = SharedMatSpec(name, D1, D2)
SharedVec(name::Symbol, D1::Int) = SharedVecSpec(name, D1)


######################
### INPUT CREATION ###
######################

@inline function create_symval_wrapper(vid::ValueId, spec::MatSpec, prog::IRProgram)
    return create_symval(vid, DenseMatKind, MatShape(spec.D1, spec.D2), prog)
end
@inline function create_symval_wrapper(vid::ValueId, spec::SharedMatSpec, prog::IRProgram)
    return create_symval(vid, SharedMatKind, MatShape(spec.D1, spec.D2), prog)
end
@inline function create_symval_wrapper(vid::ValueId, spec::VecSpec, prog::IRProgram)
    return create_symval(vid, VecKind, VecShape(spec.D1), prog)
end
@inline function create_symval_wrapper(vid::ValueId, spec::SharedVecSpec, prog::IRProgram)
    return create_symval(vid, SharedVecKind, VecShape(spec.D1), prog)
end
@inline function create_symval_wrapper(vid::ValueId, spec::ScalarSpec, prog::IRProgram)
    return create_symval(vid, ScalarKind, ScalShape(), prog)
end
@inline function create_symval_wrapper(vid::ValueId, spec::SharedScalarSpec, prog::IRProgram)
    return create_symval(vid, SharedScalarKind, ScalShape(), prog)
end

function make_input(spec::Spec, prog::IRProgram)
    vid = fresh!(prog)
    push!(prog.inputs, spec.name => vid)
    
    return create_symval_wrapper(vid, spec, prog)
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

@inline function op_signature(::Val{:mul}, ::Type{<:MatLike}, sx::MatShape, ::Type{<:MatLike}, sy::MatShape)
    sx.D2 == sy.D1 || error("Matmul shape mismatch: $sx * $sy")
    return DenseMatKind, MatShape(sx.D1, sy.D2)
end
@inline function op_signature(::Val{:mul}, ::Type{<:MatLike}, sx::MatShape, ::Type{<:VecLike}, sy::VecShape)
    sx.D2 == sy.D1 || error("Matvec multiplication shape mismatch $sx * $sy")
    return VecKind, VecShape(sx.D1)
end
@inline function op_signature(::Val{:mul}, ::Type{<:ScalLike}, ::ScalShape, ::Type{<:ScalLike}, ::ScalShape)
    error("Scalar * Scalar not yet supported")
end
@inline function op_signature(::Val{:mul}, ::Type{<:ScalLike}, ::ScalShape, ::Type{<:MatLike}, sy::MatShape)
    error("Scalar * Mat not yet supported")
end
@inline function op_signature(::Val{:mul}, ::Type{<:ScalLike}, ::ScalShape, ::Type{<:VecLike}, sy::VecShape)
    error("Scalar * Vec not yet supported")
end
@inline function op_signature(::Val{:mul}, tx::Type{<:SymKind}, ::Shape, ty::Type{<:SymKind}, ::Shape)
    error("Unknown combination of types to multiply: $tx, $ty")
end
function Base.:*(x::SymVal, y::SymVal)
    if (x.kind <: MatLike || x.kind <: VecLike) && y.kind <: ScalLike
        return y * x
    end
    kind, shape = op_signature(Val(:mul), x.kind, x.prog.shapes[x.vid], y.kind, x.prog.shapes[y.vid])
    out = emit!(x.prog, :mul, Any[x.vid, y.vid])
    return create_symval(out, kind, shape, x.prog)
end

@inline function op_signature(::Val{:mul_const}, ::Type{<:ScalLike}, sx::ScalShape, ::Type{ConstKind}, ::ConstShape)
    return ScalarKind, sx
end
@inline function op_signature(::Val{:mul_const}, ::Type{<:MatLike}, sx::MatShape, ::Type{ConstKind}, ::ConstShape)
    return DenseMatKind, sx
end
@inline function op_signature(::Val{:mul_const}, ::Type{<:VecLike}, sx::VecShape, ::Type{ConstKind}, ::ConstShape)
    return VecKind, sx
end
function Base.:*(x::SymVal, y::Number)
    kind, shape = op_signature(Val(:mul_const), x.kind, x.prog.shapes[x.vid], ConstKind, ConstShape())
    out = emit!(x.prog, :mul_const, Any[x.vid, fix_type(x.prog, y)])
    return create_symval(out, kind, shape, x.prog)
end
Base.:*(x::Number, y::SymVal) = y * x

function LinearAlgebra.cholesky(x::SymVal{SymMatKind,T}) where {T}
    out = emit!(x.prog, :chol, Any[x.vid])
    shape = x.prog.shapes[x.vid]::MatShape
    return create_choleskyval(out, shape, x.prog)
end
LinearAlgebra.cholesky(::SymVal) = error("Unknown types for Cholesky. Ensure matrix is marked as Symmetric")

function LinearAlgebra.Symmetric(x::SymVal{<:MatLike})
    shape = x.prog.shapes[x.vid]
    shape.D1 == shape.D2 || error("Cannot call Symmetric() on a matrix with shape $shape")
    out = emit!(x.prog, :sym, Any[x.vid])
    return create_symval(out, SymMatKind, shape, x.prog)
end
LinearAlgebra.Symmetric(::SymVal) = error("Symmetric() is only defined for matrices")

function LinearAlgebra.LowerTriangular(x::SymVal{<:MatLike})
    shape = x.prog.shapes[x.vid]
    shape.D1 == shape.D2 || error("Cannot call LowerTriangular() on a matrix with shape $shape")
    out = emit!(x.prog, :lowertrig, Any[x.vid])
    return create_symval(out, LowerTrigMatKind, shape, x.prog)
end
LinearAlgebra.LowerTriangular(::SymVal) = error("LowerTriangular() only defined for matrices")

function LinearAlgebra.UpperTriangular(x::SymVal{<:MatLike})
    shape = x.prog.shapes[x.vid]
    shape.D1 == shape.D2 || error("Cannot call UpperTriangular() on a matrix with shape $shape")
    out = emit!(x.prog, :uppertrig, Any[x.vid])
    return create_symval(out, UpperTrigMatKind, shape, x.prog)
end
LinearAlgebra.UpperTriangular(::SymVal) = error("UpperTriangular() only defined for matrices")

function Base.:\(x::SymVal{Kx,T}, y::SymVal{Ky,T}) where {Kx<:TrigMatKind,Ky<:MatLike,T}
    sx = x.prog.shapes[x.vid]
    sy = x.prog.shapes[y.vid]

    sx.D1 == sx.D2 || error("Triangular solve requires square L/U, got $sx")
    sx.D2 == sy.D1 || error("Leftdiv shape mismatch: $sx \\ $sy")

    op = Kx <: LowerTrigMatKind ? :forwardsolve : :backwardsolve
    out = emit!(x.prog, op, Any[x.vid, y.vid])

    return create_symval(out, DenseMatKind, sy, x.prog)
end
function Base.:\(x::SymVal{Kx,T}, y::SymVal{Ky,T}) where {Kx<:TrigMatKind,Ky<:VecLike,T}
    error("Vector triangular solves not yet supported")
end
Base.:\(::SymVal, ::SymVal) = error("Unknown combination of types to leftdiv")

function Base.:/(x::SymVal{<:MatLike}, y::SymVal{<:SymMatKind})
    sx = x.prog.shapes[x.vid]
    sy = x.prog.shapes[y.vid]
    sy.D1 == sy.D2 || error("Expected symmetric matrix to be square, got $sy")
    sx.D2 == sy.D1 || error("Matrix solve type mismatch: $sx / $sy")

    chol_res = cholesky(y)
    interm = chol_res.L \ x'
    result = chol_res.U \ interm
    return result'
end
function Base.:/(x::SymVal{<:MatLike}, y::SymVal{<:MatLike})
    error("Solves are not yet supported for non-symmetric matrices")
end
Base.:/(x::SymVal{ScalarKind}, y::SymVal{ScalarKind}) = error("Scalar/Scalar not yet supported")
Base.:/(::SymVal{<:SymKind}, ::SymVal{<:SymKind}) = error("Unknown combination of types to div")

function LinearAlgebra.qr(x::SymVal{<:MatLike})
    sx = x.prog.shapes[x.vid]
    if sx.D1 >= sx.D2
        # Cholesky QR2
        R1 = cholesky(Symmetric(x' * x)).U
        Q1 = (R1' \ x')'
        R2 = cholesky(Symmetric(Q1' * Q1)).U
        Q = (R2' \ Q1')'
        R = UpperTriangular(R2 * R1)
        return create_qrval(Q, R, x.prog)

        # Cholesky QR
        # G = BatchedKernels.gram(x)
        # R = cholesky(Symmetric(G)).U
        # Q = (LowerTriangular(R') \ x')'
        # return create_qrval(Q, R, x.prog)
    else
        error("D1 < D2 case for QR not implemented yet")
    end
end
LinearAlgebra.qr(::SymVal{<:SymKind}) = error("Unknown combination of types to qr decomposition")

function LinearAlgebra.adjoint(x::SymVal{<:MatLike})
    sx = x.prog.shapes[x.vid]
    shape = MatShape(sx.D2, sx.D1)
    out = emit!(x.prog, :trans, Any[x.vid])

    if x.prog.kinds[x.vid] <: LowerTrigMatKind
        kind = UpperTrigMatKind
    elseif x.prog.kinds[x.vid] <: UpperTrigMatKind
        kind = LowerTrigMatKind
    else
        kind = TransMatKind
    end

    return create_symval(out, kind, shape, x.prog)
end
LinearAlgebra.adjoint(x::SymVal{VecKind}) = error("Vec' not yet supported")
LinearAlgebra.adjoint(x::SymVal{ScalarKind}) = x
LinearAlgebra.adjoint(x::SymVal{<:SymKind}) = error("Unknown type to adjoint")

@inline function op_signature(::Val{:addsub}, ::Type{<:MatLike}, sx::MatShape, ::Type{<:MatLike}, sy::MatShape)
    sx == sy || error("Shape mismatch for add/sub: $sx +- $sy")
    return DenseMatKind, sx
end
@inline function op_signature(::Val{:addsub}, ::Type{<:VecLike}, sx::VecShape, ::Type{<:VecLike}, sy::VecShape)
    sx == sy || error("Shape mismatch for add/sub: $sx +- $sy")
    return VecKind, sx
end
@inline function op_signature(::Val{:addsub}, ::Type{<:ScalLike}, sx::ScalShape, ::Type{<:ScalLike}, ::ScalShape)
    return ScalarKind, sx
end
@inline function op_signature(::Val{:addsub}, ::Type{<:SymKind}, ::Shape, ::Type{<:SymKind}, ::Shape)
    error("Unknown combination of types to add/sub")
end
function Base.:+(x::SymVal, y::SymVal)
    sx = x.prog.shapes[x.vid]
    sy = x.prog.shapes[y.vid]
    kind, shape = op_signature(Val(:addsub), x.kind, sx, y.kind, sy)
    out = emit!(x.prog, :add, Any[x.vid, y.vid])
    return create_symval(out, kind, shape, x.prog)
end

function Base.:-(x::SymVal, y::SymVal)
    sx = x.prog.shapes[x.vid]
    sy = x.prog.shapes[y.vid]
    kind, shape = op_signature(Val(:addsub), x.kind, sx, y.kind, sy)
    out = emit!(x.prog, :sub, Any[x.vid, y.vid])
    return create_symval(out, kind, shape, x.prog)
end

function Base.:+(x::UniformScaling, y::SymVal{<:MatLike})
    sy = y.prog.shapes[y.vid]
    sy.D1 == sy.D2 || error("Identity + matrix only supported for square matrices")
    out = emit!(y.prog, :iplus, Any[y.vid, fix_type(y.prog, x.λ)])
    return create_symval(out, DenseMatKind, sy, y.prog)
end
Base.:+(x::SymVal, y::UniformScaling) = y + x
Base.:+(::UniformScaling, y::SymVal) = error("Unknown combination of type to add with identity")

function Base.:-(x::UniformScaling, y::SymVal{<:MatLike})
    sy = y.prog.shapes[y.vid]
    sy.D1 == sy.D2 || error("Identity - matrix only supported for square matrices")
    out = emit!(y.prog, :iminus, Any[y.vid, fix_type(y.prog, x.λ)])
    return create_symval(out, DenseMatKind, sy, y.prog)
end
Base.:-(x::SymVal{<:MatLike}, y::UniformScaling) = (-y) + x
Base.:-(::UniformScaling, y::SymVal) = error("Unknown combination of type to sub with identity")

Base.:+(x::SymVal) = x

Base.:-(x::SymVal) = fix_type(x.prog, -1) * x

"Normalise return value to Vector{SymVal}."
function as_symvals(out)
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
Trace a function f on symbolic inputs specified by specs.
"""
function trace(f, specs::Tuple; T::Type, D::Int, nthreads::Int)
    prog = IRProgram(T, D, nthreads)

    syms = map(s -> make_input(s, prog), specs)

    out = f(syms...)
    outs = as_symvals(out)

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