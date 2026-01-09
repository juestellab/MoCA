"""
    pad_array(A::Array)
Add a single pixel at each boundary of `A`. Those added pixels are filled with the value of their interior neighbour.
"""
function pad_array(A::Array)
    A_padded = Array{eltype(A)}(undef, (size(A) .+ 2)...)

    # Inner Volume
    A_padded[2:(end-1), 2:(end-1), 2:(end-1)] .= A

    # Faces
    A_padded[1, 2:(end-1), 2:(end-1)] .= A[1, :, :]
    A_padded[2:(end-1), 1, 2:(end-1)] .= A[:, 1, :]
    A_padded[2:(end-1), 2:(end-1), 1] .= A[:, :, 1]
    A_padded[end, 2:(end-1), 2:(end-1)] .= A[end, :, :]
    A_padded[2:(end-1), end, 2:(end-1)] .= A[:, end, :]
    A_padded[2:(end-1), 2:(end-1), end] .= A[:, :, end]

    # Edges
    A_padded[1, 1, 2:(end-1)] .= A[1,1,:]
    A_padded[1, 2:(end-1), 1] .= A[1,:,1]
    A_padded[2:(end-1), 1, 1] .= A[:,1,1]
    A_padded[1, end, 2:(end-1)] .= A[1,end,:]
    A_padded[1, 2:(end-1), end] .= A[1,:,end]
    A_padded[end, 1, 2:(end-1)] .= A[end,1,:]
    A_padded[end, 2:(end-1), 1] .= A[end,:,1]
    A_padded[2:(end-1), 1, end] .= A[:,1,end]
    A_padded[2:(end-1), end, 1] .= A[:,end,1]
    A_padded[end, end, 2:(end-1)] .= A[end,end,:]
    A_padded[end, 2:(end-1), end] .= A[end,:,end]
    A_padded[2:(end-1), end, end] .= A[:,end,end]

    # Corners
    A_padded[1,1,1] = A[1,1,1]
    A_padded[1,1,end] = A[1,1,end]
    A_padded[1,end,1] = A[1,end,1]
    A_padded[end,1,1] = A[end,1,1]
    A_padded[1,end,end] = A[1,end,end]
    A_padded[end,1,end] = A[end,1,end]
    A_padded[end,end,1] = A[end,end,1]
    A_padded[end,end,end] = A[end,end,end]

    return A_padded
end

"""
    unpad(x)
Remove padding at boundary of `x`
"""
unpad(x) = Array(x)[2:(end-1), 2:(end-1), 2:(end-1)]

"""
    cuda_params(N, n_threads, n_blocks)
For `N` total kernel evaluations and a target of `n_threads` threads and `n_blocks` blocks, compute the number of batches required to reach at least `N` kernel evaluations.
In case `n_threads * n_blocks > N`, those parameters are adapted to evaluate only the smallest number of required blocks.
"""
function cuda_params(N, n_threads, n_blocks)
    if N < n_threads * n_blocks 
        n_threads = (N < n_threads) ? N : n_threads
        n_blocks = N ÷ n_threads
    end
    n_batch = n_threads * n_blocks
    return n_threads, n_blocks, n_batch
end
