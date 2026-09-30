@testitem "Test discovery stays within test directory" tags = [:cpu] begin
    using BatchedKernels

    mktempdir() do root
        tests = joinpath(root, "test")
        nested = joinpath(root, ".claude", "worktrees", "other", "test")
        mkpath(joinpath(tests, "subdir"))
        mkpath(nested)
        runner = joinpath(tests, "runtests.jl")
        cp(joinpath(pkgdir(BatchedKernels), "test", "runtests.jl"), runner)
        markers = joinpath(root, "markers.txt")
        for (folder, name) in ((tests, "root"), (joinpath(tests, "subdir"), "subdir"))
            write(
                joinpath(folder, "items.jl"),
                """
                @testitem "$name" tags=[:cpu] begin
                    open($(repr(markers)), "a") do io
                        println(io, "$name")
                    end
                    @test true
                end
                """,
            )
        end
        write(
            joinpath(nested, "items.jl"),
            """
            @testitem "nested checkout must not run" tags=[:cpu] begin
                error("discovered a sibling worktree")
            end
            """,
        )
        withenv("BATCHEDKERNELS_TEST_CPU_ONLY" => "true") do
            Base.include(Module(gensym(:RunnerFixture)), runner)
        end
        @test sort(readlines(markers)) == ["root", "subdir"]
    end
end
