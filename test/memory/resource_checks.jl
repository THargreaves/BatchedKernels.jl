# Shared test helper: inspect the final device binary, not just pre-assembly PTX.
function register_resources(kernel)
    memory = CUDA.memory(kernel)
    return (
        registers=CUDA.registers(kernel),
        local_bytes=getproperty(memory, :local),
        shared_bytes=memory.shared,
    )
end

function require_register_resident(kernel)
    resources = register_resources(kernel)
    resources.local_bytes == 0 || error(
        "Expected register-resident kernel, found $(resources.local_bytes) local-memory bytes per thread",
    )
    return resources
end
