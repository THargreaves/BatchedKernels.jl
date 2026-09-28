@testitem "Composite parameters preserve independent matrix shapes" tags = [:cpu] begin
    using BatchedKernels, LinearAlgebra
    const BK = BatchedKernels
    struct Observation{H,C,R}
        H::H
        c::C
        R::R
    end
    struct Repeated{M}
        A::M
        B::M
    end
    H = BatchedCuMatrix(zeros(Float32, 2, 3, 5))
    c = BatchedCuVector(zeros(Float32, 2, 5))
    R = BatchedCuMatrix(zeros(Float32, 2, 2, 5))
    components = (; H, c, R)
    OT = Observation{eltype(H),eltype(c),eltype(R)}
    @test eltype(H) === eltype(R)
    observation = BatchedStruct{OT,typeof(components)}(components, 5)
    TT = BK.trace_element_type(typeof(observation))
    @test fieldtype(TT, :H) === BK.TraceMatrix{Float32,2,3}
    @test fieldtype(TT, :R) === BK.TraceMatrix{Float32,2,2}
    f(o) = o.H * o.H' + o.R
    tape = BK.trace(f, BK.InputSpec[BK.input_spec(observation)])
    @test BK.shape(tape.metas[tape.output.id].type) == (2, 2)
    # Distinct symbolic parameters remain distinct even when concrete values match.
    S = Observation{Matrix{Float32},Vector{Float32},Matrix{Float32}}
    replaced = BK._replace_composite_field_types(
        S, Dict(:H => Matrix{Float32}, :R => UpperTriangular{Float32,Matrix{Float32}})
    )
    @test fieldtype(replaced, :H) === Matrix{Float32}
    @test fieldtype(replaced, :R) === UpperTriangular{Float32,Matrix{Float32}}
    @test_throws ArgumentError BK._replace_composite_field_types(
        Repeated{Matrix{Float32}},
        Dict(:A => BK.TraceMatrix{Float32,2,3}, :B => BK.TraceMatrix{Float32,2,2}),
    )
end
