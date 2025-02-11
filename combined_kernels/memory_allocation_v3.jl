using CUDA
using CUDA: i32

# Type computation methods for batch operations
batch_mul(::Type{Matrix}, ::Type{Matrix}) = Matrix
batch_mul(::Type{Matrix}, ::Type{Vector}) = Vector
batch_add(::Type{Matrix}, ::Type{Matrix}) = Matrix
batch_add(::Type{Vector}, ::Type{Vector}) = Vector
batch_sub(::Type{Vector}, ::Type{Vector}) = Vector
batch_sub(::Type{Matrix}, ::Type{Matrix}) = Matrix
batch_chol(::Type{Matrix}) = Matrix
batch_solve(::Type{Matrix}, ::Type{Matrix}) = Matrix
batch_solve(::Type{Matrix}, ::Type{Vector}) = Vector

# Structures to represent the computation DAG
struct DAGNode
    op::Symbol  # Operation (e.g., :batch_mul, :batch_add)
    inputs::Vector{Any}  # Input nodes or variable names
    options::Dict{Symbol,Any}  # Options like trans_B=true
    output_type::Type  # Output type (Matrix or Vector)
end

struct ComputationDAG
    nodes::Vector{DAGNode}
    intermediate_vars::Dict{Symbol,DAGNode}  # Maps variable names to nodes
    var_types::Dict{Symbol,Type}  # Maps variable names to types
    outputs::Vector{Symbol}  # Final output variables
end

# Compute output type for a node given operation and input types
function compute_node_type(op::Symbol, input_types::Vector{Type}, options::Dict{Symbol,Any})
    if op === :batch_mul
        return batch_mul(input_types[1], input_types[2])
    elseif op === :batch_add
        return batch_add(input_types[1], input_types[2])
    elseif op === :batch_sub
        return batch_sub(input_types[1], input_types[2])
    elseif op === :batch_chol
        return batch_chol(input_types[1])
    elseif op === :batch_solve
        return batch_solve(input_types[1], input_types[2])
    else
        error("Unsupported operation: $op")
    end
end

# Get type of an input (either DAGNode or variable)
function get_input_type(
    input::Any, var_types::Dict{Symbol,Type}, intermediate_vars::Dict{Symbol,DAGNode}
)
    if isa(input, DAGNode)
        return input.output_type
    elseif isa(input, Symbol)
        if haskey(var_types, input)
            return var_types[input]
        else
            error("Unknown variable: $input")
        end
    else
        error("Unexpected input type: $(typeof(input))")
    end
end

# Process an expression to create DAG nodes
function process_expr(
    expr, var_types::Dict{Symbol,Type}, intermediate_vars::Dict{Symbol,DAGNode}
)
    if isa(expr, Symbol)
        return haskey(intermediate_vars, expr) ? intermediate_vars[expr] : expr
    elseif isa(expr, Expr)
        if expr.head == :call
            op = expr.args[1]
            if startswith(string(op), "batch_")
                # Extract regular arguments and keyword arguments
                regular_args = []
                kwargs = Dict{Symbol,Any}()

                # Check for keyword arguments (parameters)
                if length(expr.args) >= 2 &&
                    isa(expr.args[2], Expr) &&
                    expr.args[2].head == :parameters
                    for kwarg in expr.args[2].args
                        if isa(kwarg, Expr) && kwarg.head == :kw
                            kwargs[kwarg.args[1]] = kwarg.args[2]
                        end
                    end
                    # Process remaining arguments after the parameters
                    for arg in expr.args[3:end]
                        push!(regular_args, process_expr(arg, var_types, intermediate_vars))
                    end
                else
                    # No keyword arguments, process all arguments normally
                    for arg in expr.args[2:end]
                        push!(regular_args, process_expr(arg, var_types, intermediate_vars))
                    end
                end

                input_types = Type[
                    get_input_type(arg, var_types, intermediate_vars) for
                    arg in regular_args
                ]
                output_type = compute_node_type(op, input_types, kwargs)

                return DAGNode(op, regular_args, kwargs, output_type)
            end
        end
    end
    return error("Unsupported expression: $expr")
end

# Macro to create DAG from function definition
macro create_dag(expr)
    if expr.head != :function
        error("Expression must be a function definition")
    end

    # Extract function arguments and their types
    func_args = expr.args[1].args[2:end]
    var_types = Dict{Symbol,Type}()
    for arg in func_args
        if isa(arg, Expr) && arg.head == :(::)
            var_types[arg.args[1]] = eval(arg.args[2])
        end
    end

    # Process function body
    body = expr.args[2]
    if body.head == :block
        nodes = []
        intermediate_vars = Dict{Symbol,DAGNode}()
        outputs = []

        for stmt in body.args
            if isa(stmt, Expr)
                if stmt.head == :(=)
                    rhs = process_expr(stmt.args[2], var_types, intermediate_vars)
                    if isa(rhs, DAGNode)
                        push!(nodes, rhs)
                        intermediate_vars[stmt.args[1]] = rhs
                        var_types[stmt.args[1]] = rhs.output_type
                    end
                elseif stmt.head == :return
                    if isa(stmt.args[1], Expr) && stmt.args[1].head == :tuple
                        append!(
                            outputs, [arg for arg in stmt.args[1].args if isa(arg, Symbol)]
                        )
                    elseif isa(stmt.args[1], Symbol)
                        push!(outputs, stmt.args[1])
                    end
                end
            end
        end

        return :(ComputationDAG($nodes, $intermediate_vars, $var_types, $outputs))
    end
end

using Graphs
using GraphPlot
using Compose
using Cairo
using Fontconfig

# Convert DAG to interference graph for coloring
function to_interference_graph(dag::ComputationDAG)
    # Create sets of nodes for matrices and vectors
    matrix_nodes = Set{Symbol}()
    vector_nodes = Set{Symbol}()

    # Add input variables
    for (var, type) in dag.var_types
        if type == Matrix
            push!(matrix_nodes, var)
        else
            push!(vector_nodes, var)
        end
    end

    # Create interference edges
    matrix_edges = Set{Tuple{Symbol,Symbol}}()
    vector_edges = Set{Tuple{Symbol,Symbol}}()

    # Create a mapping from DAGNode to its output variable name
    node_to_var = Dict(node => var for (var, node) in dag.intermediate_vars)

    # Process each operation to add interference edges
    for (var, node) in dag.intermediate_vars
        # Add node to appropriate set
        if node.output_type == Matrix
            push!(matrix_nodes, var)
        else
            push!(vector_nodes, var)
        end

        # Get all input variables and intermediate nodes
        all_inputs = []
        input_vars = Symbol[]
        for input in node.inputs
            if isa(input, Symbol)
                push!(input_vars, input)
                push!(all_inputs, input)
            elseif isa(input, DAGNode)
                push!(all_inputs, node_to_var[input])
            end
        end

        # Add edges between all pairs of inputs (including intermediate nodes)
        for i in 1:length(all_inputs)
            for j in (i + 1):length(all_inputs)
                input_i, input_j = all_inputs[i], all_inputs[j]
                if dag.var_types[input_i] == Matrix && dag.var_types[input_j] == Matrix
                    push!(matrix_edges, minmax(input_i, input_j))
                elseif dag.var_types[input_i] == Vector && dag.var_types[input_j] == Vector
                    push!(vector_edges, minmax(input_i, input_j))
                end
            end
            # Add edge between each input and the output if same type
            if dag.var_types[all_inputs[i]] == node.output_type
                if node.output_type == Matrix
                    push!(matrix_edges, minmax(var, all_inputs[i]))
                else
                    push!(vector_edges, minmax(var, all_inputs[i]))
                end
            end
        end
    end

    return matrix_nodes, vector_nodes, matrix_edges, vector_edges
end

# Compute memory allocations from interference graph
function compute_memory_allocation(dag::ComputationDAG)
    matrix_nodes, vector_nodes, matrix_edges, vector_edges = to_interference_graph(dag)

    # Create interference graphs for matrices and vectors
    matrix_list = collect(matrix_nodes)
    vector_list = collect(vector_nodes)

    # Matrix graph
    g_matrix = SimpleGraph(length(matrix_list))
    matrix_to_idx = Dict(node => i for (i, node) in enumerate(matrix_list))
    for (u, v) in matrix_edges
        add_edge!(g_matrix, matrix_to_idx[u], matrix_to_idx[v])
    end

    # Vector graph
    g_vector = SimpleGraph(length(vector_list))
    vector_to_idx = Dict(node => i for (i, node) in enumerate(vector_list))
    for (u, v) in vector_edges
        add_edge!(g_vector, vector_to_idx[u], vector_to_idx[v])
    end

    # Color both graphs, handling empty cases
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    matrix_allocation = Dict{Symbol,String}()
    if !isempty(matrix_nodes)
        matrix_colors = Graphs.random_greedy_color(g_matrix, 1000)
        for var in matrix_nodes
            matrix_allocation[var] = palette[matrix_colors.colors[matrix_to_idx[var]]]
        end
    end

    vector_allocation = Dict{Symbol,String}()
    if !isempty(vector_nodes)
        vector_colors = Graphs.random_greedy_color(g_vector, 1000)
        for var in vector_nodes
            vector_allocation[var] = palette[vector_colors.colors[vector_to_idx[var]]]
        end
    end

    return matrix_allocation, vector_allocation
end

# Convert DAG to graph for visualization
function to_graph(dag::ComputationDAG)
    g = DiGraph()
    node_to_idx = Dict{Union{Symbol,DAGNode},Int}()
    idx_to_label = Dict{Int,String}()
    current_idx = 1

    # Helper function to collect all nodes in computation flow
    function collect_nodes(node::DAGNode, nodes_set)
        push!(nodes_set, node)
        for input in node.inputs
            if isa(input, DAGNode)
                collect_nodes(input, nodes_set)
            end
        end
    end

    # Collect all nodes and input variables
    computation_nodes = Set{DAGNode}()
    for output in dag.outputs
        collect_nodes(dag.intermediate_vars[output], computation_nodes)
    end

    input_vars = Set{Symbol}()
    for node in computation_nodes
        for input in node.inputs
            if isa(input, Symbol)
                push!(input_vars, input)
            end
        end
    end

    # Add vertices for input variables
    for var in input_vars
        add_vertex!(g)
        node_to_idx[var] = current_idx
        idx_to_label[current_idx] = string(var)
        current_idx += 1
    end

    # Add vertices for computation nodes
    for node in computation_nodes
        add_vertex!(g)
        var = first([k for (k, v) in dag.intermediate_vars if v === node])
        node_to_idx[node] = current_idx
        idx_to_label[current_idx] = string(var)
        current_idx += 1
    end

    # Add edges
    for node in computation_nodes
        target_idx = node_to_idx[node]
        for input in node.inputs
            if isa(input, Symbol)
                source_idx = node_to_idx[input]
                add_edge!(g, source_idx, target_idx)
            elseif isa(input, DAGNode)
                source_idx = node_to_idx[input]
                add_edge!(g, source_idx, target_idx)
            end
        end
    end

    return g, idx_to_label
end

# Plot DAG with memory allocation colors
function plot_dag(dag::ComputationDAG)
    # Get memory allocations
    matrix_allocation, vector_allocation = compute_memory_allocation(dag)

    # Create graph
    g, idx_to_label = to_graph(dag)

    # Prepare node colors and labels
    labels = [idx_to_label[i] for i in 1:nv(g)]
    fill_colors = String[]
    stroke_colors = String[]

    # Set colors based on memory allocation
    for i in 1:nv(g)
        var = Symbol(idx_to_label[i])
        var_type = dag.var_types[var]

        if var_type == Matrix
            push!(fill_colors, matrix_allocation[var])
            push!(stroke_colors, matrix_allocation[var])
        else  # Vector
            push!(fill_colors, "white")
            push!(stroke_colors, vector_allocation[var])
        end
    end

    # Create layout
    locs_x, locs_y = spring_layout(g; C=4.0, MAXITER=1000)

    # Plot
    p = gplot(
        g,
        locs_x,
        locs_y;
        nodelabel=labels,
        nodefillc=fill_colors,
        nodestrokec=stroke_colors,
        NODELABELSIZE=6.0,
        nodesize=0.8,
        arrowlengthfrac=0.04,
        nodestrokelw=5.0,
        background_color="white",
        font_family="Fira Sans",
    )

    draw(PNG("dag.png", 32cm, 32cm), p)
    return matrix_allocation, vector_allocation
end

# Example usage
# dag = @create_dag function kalman_predict(A::Matrix, P::Matrix, Q::Matrix, m::Vector, b::Vector)
#     AP = batch_mul(A, P)
#     APA = batch_mul(AP, A; trans_B=true)
#     P_new = batch_add(APA, Q)
#     Am = batch_mul(A, m)
#     m_new = batch_add(Am, b)
#     return P_new, m_new
# end

dag = @create_dag function four_colour(A::Matrix, B::Matrix)
    C = batch_mul(A, B)
    D = batch_mul(B, A)
    E = batch_mul(C, D)

    return E
end

# Example requiring 5 slots
dag = @create_dag function five_colour(A::Matrix, B::Matrix)
    C = batch_mul(A, B)
    D = batch_mul(B, A)
    E = batch_mul(C, D)
    F = batch_mul(E, A)
    G = batch_mul(E, B)
    H = batch_add(F, G)
    return H
end

# dag = @create_dag function kalman_step(
#     A::Matrix, Σ::Matrix, Q::Matrix, μ::Vector, b::Vector, y::Vector, H::Matrix, R::Matrix
# )
#     # TODO: allow these to be temporary variables
#     AΣ = batch_mul(A, Σ)
#     AΣAT = batch_mul(AΣ, A; trans_B=true)  # NOTE: not kwarg currently
#     Σ̃ = batch_add(AΣAT, Q)
#     Aμ = batch_mul(A, μ)
#     μ̃ = batch_add(Aμ, b)

#     ỹ = batch_mul(H, μ̃)
#     ε = batch_sub(y, ỹ)

#     Σ̃HT = batch_mul(H, Σ̃; trans_B=true)
#     HΣ̃HT = batch_mul(H, Σ̃HT)
#     S = batch_add(HΣ̃HT, R)
#     L = batch_chol(S)
#     K = batch_solve(Σ̃, L)

#     Kε = batch_mul(K, ε)
#     μ̂ = batch_add(μ̃, Kε)

#     KH = batch_mul(K, H)
#     # Ignore 1 - I for now
#     Σ̂ = batch_mul(KH, Σ̃)

#     return Σ̂, μ̂
# end

matrix_alloc, vector_alloc = plot_dag(dag)

# Find input and output variable positions
function analyze_io_positions(dag::ComputationDAG)
    # Get list of input variables (those not in intermediate_vars)
    inputs = Symbol[]
    for (var, _) in dag.var_types
        if !haskey(dag.intermediate_vars, var)
            push!(inputs, var)
        end
    end

    # Create mappings
    input_positions = Dict(var => i for (i, var) in enumerate(sort(inputs)))
    output_positions = Dict(var => i for (i, var) in enumerate(dag.outputs))

    return input_positions, output_positions
end

# Shared memory version of batch_mul for use in kernel
# Can't have $ here since not in a quote. Maybe we just put this in a quote block?
# function batch_mul_shmem!(
#     out_slot::Int,
#     A_slot::Int,
#     B_slot::Int,
#     shmem_matrices::CuDeviceArray{T,4},  # (D, D, N_B, M_M)
#     thread_row::Int32,
#     thread_block_mtrx::Int32,
# ) where {T}
#     if thread_row <= $D
#         # Each thread computes one row of the output
#         for j in 1:($D)
#             result = zero(T)
#             for k in 1:($D)
#                 result +=
#                     shmem_matrices[thread_row, k, thread_block_mtrx, A_slot] *
#                     shmem_matrices[k, j, thread_block_mtrx, B_slot]
#             end
#             shmem_matrices[thread_row, j, thread_block_mtrx, out_slot] = result
#         end
#     end
#     return nothing
# end

# Helper functions for kernel
function compute_warp_params(D::Int)
    N_W = div(32, D)  # matrices per warp
    N_threads = 32    # threads per warp
    return N_W, N_threads
end

# Find first and last use of each variable
function analyze_variable_usage(dag::ComputationDAG)
    first_use = Dict{Symbol,Int}()
    last_use = Dict{Symbol,Int}()

    # Initialize inputs
    for (var, _) in dag.var_types
        if !haskey(dag.intermediate_vars, var)
            first_use[var] = typemax(Int)
            last_use[var] = 0
        end
    end

    # Process each operation in order
    for (op_idx, node) in enumerate(dag.nodes)
        # Find variable name for this node's output
        out_var = first([k for (k, v) in dag.intermediate_vars if v === node])

        # Process inputs
        for input in node.inputs
            var = if isa(input, Symbol)
                input
            else  # DAGNode
                first([k for (k, v) in dag.intermediate_vars if v === input])
            end

            # Update first/last use
            first_use[var] = min(get(first_use, var, op_idx), op_idx)
            last_use[var] = max(get(last_use, var, op_idx), op_idx)
        end

        # Add output variable
        first_use[out_var] = op_idx
        last_use[out_var] = op_idx
    end

    # Update last use for output variables
    for output in dag.outputs
        last_use[output] = typemax(Int)  # Keep until end
    end

    return first_use, last_use
end

# Kernel generation
function generate_batched_kernel(dag::ComputationDAG, D::Int, N_B::Int)
    # Get memory allocations and convert colors to slot indices
    matrix_alloc, _ = compute_memory_allocation(dag)
    # Create mapping of colors to indices
    color_to_slot = Dict(
        color => i for (i, color) in enumerate(unique(values(matrix_alloc)))
    )
    # Convert allocation dictionary to use slot numbers
    matrix_slots = Dict(var => color_to_slot[color] for (var, color) in matrix_alloc)
    M_M = length(color_to_slot)  # Number of matrix memory slots needed

    # Analyze variable usage
    first_use, last_use = analyze_variable_usage(dag)

    # Create the kernel function
    return quote
        function batched_kernel!(
            outputs::NTuple{N_out,CuDeviceArray{T,3}},  # Each output is (D, D, N)
            inputs::NTuple{N_in,CuDeviceArray{T,3}},    # Each input is (D, D, N)
            N::Int32,
        ) where {T,N_out,N_in}
            # Thread indexing
            tid = threadIdx().x
            bid = blockIdx().x
            warp_id = div(tid - 1i32, 32) + 1i32
            lane_id = mod1(tid, 32)

            # Matrix indexing within warp/block
            N_W = $(compute_warp_params(D)[1])
            warp_mtrx_id = div(lane_id - 1i32, $D) + 1i32
            thread_row = mod1(lane_id, $D)
            thread_block_mtrx = (warp_id - 1i32) * N_W + warp_mtrx_id
            grid_mtrx_id = (bid - 1i32) * $N_B + thread_block_mtrx

            # Shared memory allocation
            shmem_matrices = @cuStaticSharedMem(T, ($D, $D, $N_B, $M_M))

            # Generate sequence of operations with interleaved loads
            $(generate_operation_sequence(
                dag, matrix_slots, first_use, last_use, analyze_io_positions(dag)...
            ))

            return nothing
        end
    end
end

# Helper to generate a load for a variable
function generate_load(var::Symbol, slot::Int, input_idx::Int)
    return quote
        # if tid == 255 && bid == 10
        #     @cuprintln grid_mtrx_id
        #     @cuprintln thread_block_mtrx
        # end
        if grid_mtrx_id <= N && warp_mtrx_id <= N_W
            for j in 1:($D)
                for i in 1:($D)
                    if thread_row == i
                        shmem_matrices[i, j, thread_block_mtrx, $slot] = inputs[$input_idx][
                            i, j, grid_mtrx_id
                        ]
                    end
                end
            end
        end
        # if tid == 1 && bid == 1
        #     @cuprintln shmem_matrices[1, 1, thread_block_mtrx, $slot]
        # end
    end
end

# Helper to generate a store for a variable
function generate_store(var::Symbol, slot::Int, output_idx::Int)
    return quote
        # if tid == 255 && bid == 10
        #     @cuprintln grid_mtrx_id
        #     @cuprintln thread_block_mtrx
        # end
        if grid_mtrx_id <= N && warp_mtrx_id <= N_W
            for j in 1:($D)
                for i in 1:($D)
                    if thread_row == i
                        outputs[$output_idx][i, j, grid_mtrx_id] = shmem_matrices[
                            i, j, thread_block_mtrx, $slot
                        ]
                    end
                end
            end
        end
    end
end

# Helper to generate operation sequence
function generate_operation_sequence(
    dag::ComputationDAG,
    matrix_slots::Dict{Symbol,Int},
    first_use::Dict{Symbol,Int},
    last_use::Dict{Symbol,Int},
    input_positions::Dict{Symbol,Int},
    output_positions::Dict{Symbol,Int},
)
    sequence = Expr(:block)

    # Process each operation
    for (op_idx, node) in enumerate(dag.nodes)
        # First, load any inputs that are needed for the first time
        for input in node.inputs
            if isa(input, Symbol) && first_use[input] == op_idx
                slot = matrix_slots[input]
                input_idx = input_positions[input]
                push!(sequence.args, generate_load(input, slot, input_idx))
            end
        end

        if node.op === :batch_mul
            # Get memory slots for inputs and output
            input1 = first(node.inputs)
            input2 = node.inputs[2]
            output = first([k for (k, v) in dag.intermediate_vars if v === node])

            slot1 = matrix_slots[if input1 isa Symbol
                input1
            else
                first([k for (k, v) in dag.intermediate_vars if v === input1])
            end]
            slot2 = matrix_slots[if input2 isa Symbol
                input2
            else
                first([k for (k, v) in dag.intermediate_vars if v === input2])
            end]
            out_slot = matrix_slots[output]

            # Add the multiplication
            push!(
                sequence.args,
                quote
                    if grid_mtrx_id <= N && warp_mtrx_id <= N_W
                        if thread_row <= $D
                            for j in 1:($D)
                                result = zero(T)
                                for k in 1:($D)
                                    result +=
                                        shmem_matrices[
                                            thread_row, k, thread_block_mtrx, $slot1
                                        ] * shmem_matrices[k, j, thread_block_mtrx, $slot2]
                                end
                                shmem_matrices[
                                    thread_row, j, thread_block_mtrx, $out_slot
                                ] = result
                            end
                        end
                    end
                    # if tid == 1 && bid == 1
                    #     @cuprintln shmem_matrices[1, 1, thread_block_mtrx, $out_slot]
                    # end
                end,
            )
        end

        # Store any results that won't be used again
        for input in node.inputs
            var = if isa(input, Symbol)
                input
            else
                first([k for (k, v) in dag.intermediate_vars if v === input])
            end

            if last_use[var] == op_idx && haskey(output_positions, var)
                slot = matrix_slots[var]
                push!(sequence.args, generate_store(var, slot, output_positions[var]))
            end
        end

        # Store output if it's a final result and won't be used again
        output = first([k for (k, v) in dag.intermediate_vars if v === node])
        if haskey(output_positions, output) &&
            (last_use[output] == op_idx || last_use[output] == typemax(Int))
            slot = matrix_slots[output]
            push!(sequence.args, generate_store(output, slot, output_positions[output]))
        end
    end

    return sequence
end

# Test kernel
D = 3
N = 10000
n_mats_per_warp = floor(Int, 32 / D)
n_threads = 256
n_warps_per_block = div(n_threads, 32)
n_mats_per_block = n_mats_per_warp * n_warps_per_block
n_blocks = ceil(Int, N / n_mats_per_block)

A_test = CUDA.rand(Float32, D, D, N)
B_test = CUDA.rand(Float32, D, D, N)
C_test = CUDA.zeros(Float32, D, D, N)
E_test = CUDA.zeros(Float32, D, D, N)

dag_test = @create_dag function four_colour(A::Matrix, B::Matrix)
    C = batch_mul(A, B)
    D = batch_mul(B, A)
    E = batch_mul(C, D)
    return C, E
end

kernel_expr = generate_batched_kernel(dag_test, D, n_mats_per_block)
kernel_func = eval(kernel_expr)

@cuda threads = n_threads blocks = n_blocks kernel_func(
    (C_test, E_test), (A_test, B_test), Int32(N)
)

C_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', A_test, B_test)
D_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', B_test, A_test)
E_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', C_val, D_val)

println("C error: ", CUDA.norm(C_test - C_val))
println("E error: ", CUDA.norm(E_test - E_val))

# Benchmark
using BenchmarkTools

N_bench = 10^6

@benchmark (CUDA.@sync begin
    C_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', A_test, B_test)
    D_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', B_test, A_test)
    E_val = CUDA.CUBLAS.gemm_strided_batched('N', 'N', C_val, D_val)
end) setup = begin
    A_test = CUDA.rand(Float32, $D, $D, $N_bench)
    B_test = CUDA.rand(Float32, $D, $D, $N_bench)
end

@benchmark (CUDA.@sync begin
    C_test = CUDA.zeros(Float32, $D, $D, $N_bench)
    E_test = CUDA.zeros(Float32, $D, $D, $N_bench)
    @cuda threads = $n_threads blocks = $n_blocks kernel_func(
        (C_test, E_test), (A_test, B_test), Int32($N_bench)
    )
end) setup = begin
    A_test = CUDA.rand(Float32, $D, $D, $N_bench)
    B_test = CUDA.rand(Float32, $D, $D, $N_bench)
end
