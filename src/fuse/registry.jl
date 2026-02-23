using CUDA: i32

const LOCK = ReentrantLock()
const NEXT_PID = Threads.Atomic{UInt64}(1)

@inline function _fused_kernel(::Val{pid}, args...) where {pid}
    if threadIdx().x == 1i32 && blockIdx().x == 1i32
        @cuprintln("Fallback kernel reached, unregistered pid")
    end

    return nothing
end

function register_program!(prog::BatchedKernels.IRProgram, debug::Bool)::UInt64
    pid = Threads.atomic_add!(NEXT_PID, UInt64(1))

    num_outputs = length(prog.outputs)
    num_inputs = length(prog.inputs)
    out_syms = [Symbol(:_out, i) for i in 1:num_outputs]
    in_syms = [Symbol(:_in, i) for i in 1:num_inputs]

    body = BatchedKernels.emit_kernel_expr(
        prog;
        output_names=out_syms,
        input_names=in_syms,
    )

    lock(LOCK) do
        if debug
            println(prog)
            println("Kernel code:")
            debug_print_expr(body)
        end

        args = [
            out_syms...,
            in_syms...,
            :(N::Int32),
        ]

        Core.eval(BatchedKernels, quote
            @inline function _fused_kernel(::Val{$pid}, $(args...))
                $(body)
            end
        end)
    end

    return pid
end