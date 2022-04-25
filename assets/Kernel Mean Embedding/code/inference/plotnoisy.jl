# This file was generated, do not modify it. # hide
seed!(78)
	X = noisy_circle(100, 20, 1)
	fignoisy = scatter(X[:,1], X[:,2])
    savefig(joinpath(@OUTPUT, "fignoisy.svg")) # hide