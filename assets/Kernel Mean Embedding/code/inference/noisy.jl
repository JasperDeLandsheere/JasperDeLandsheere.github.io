# This file was generated, do not modify it. # hide
function noisy_circle(n, R, noise)
    t = 2π * rand(n)
	X = R .* [cos.(t) sin.(t)] .+ noise .* randn(n, 2)
    return X