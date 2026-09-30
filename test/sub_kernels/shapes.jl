@testmodule SubKernelShapes begin
    # Each sub-kernel is compiled per shape, so sweeping every size costs one GPU
    # compilation per combination. Matrices padded to size D are packed 32 ÷ D per
    # warp: 2 and 8 fill a warp exactly, 3 and 5 leave idle lanes, and `max_dim` is
    # the largest size a kernel family supports.
    square(max_dim) = (2, 3, 5, 8, max_dim)

    # Tall and wide pairs whose padded sizes cover the same packings.
    rectangular(max_dim) = ((2, 5), (5, 2), (3, 8), (8, 3), (4, max_dim), (max_dim, 4))

    # Logical sizes D1 inside padded sizes D ≥ D1, with and without padding.
    padded(max_dim) = ((2, 2), (2, 5), (3, 8), (8, 8), (4, max_dim), (max_dim, max_dim))
end
