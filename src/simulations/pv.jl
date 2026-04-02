using Random, CUDA, HDF5, ArgParse, ProgressMeter
import OrdinaryDiffEq as ODE
import DiffEqCallbacks as CB, NonlinearSolve as NLS
include("wzeta.jl")

# Parse argument
# aps = ArgParseSettings()
# @add_arg_table! aps begin
# 	"--npoints", "-n"
# 	help = "Number of point vortices (even number, half positive halfnegative)."
# 	arg_type = Int
# 	default = 10
# 	"--tspan"
# 	help = "Time span of the simulation"
# 	arg_type = Float64
# 	default = 100
# 	"--dt"
# 	help = "Time step size."
# 	arg_type = Float64
# 	default = 0.01
# 	"--gpu"
# 	help = "Using GPU acceleration."
# 	action = :store_true
# 	"--path", "-p"
# 	help = "The path of the output file."
# 	arg_type = String
# 	default = @__DIR__
# end
# args = parse_args(aps)
# npoints = args["npoints"]
# tspan = (0, args["tspan"])
# dt = args["dt"]
# device_array = args["gpu"] ? CuArray : Array

npoints = 10
tspan = (0, 100)
dt = 0.01


Γ = [ones(npoints÷2); -ones(npoints÷2)]

odefunc!(du, u, p, t) = begin
	x, y = u[1:(end÷2)], u[(end÷2+1):end]
	du .= vec(vcat(velocity(x, y, Γ)...))
end


u₀ = rand(2 * npoints)

initial_hamiltonian = begin
	x, y = u₀[1:(end÷2)], u₀[(end÷2+1):end]
	hamiltonian(x, y, Γ)
end

function isoHamiltonian_manifold(residual, u, p, t)
	x, y = u[1:(end÷2)], u[(end÷2+1):end]
	residual .= initial_hamiltonian - hamiltonian(x, y, Γ)
end


cb = CB.ManifoldProjection(isoHamiltonian_manifold, autodiff = NLS.AutoForwardDiff())

prob = ODE.ODEProblem(odefunc!, u₀, tspan)

sol = ODE.solve(prob, ODE.RK4(), dt = dt, callback = cb, adaptive = false)

using Plots

plot(sol, idxs = (1,11))
plot(sol, idxs = (2,12))
plot(sol, idxs = (3,13))
plot(sol, idxs = (4,14))


H = map(sol.u) do u
    x, y = u[1:(end÷2)], u[(end÷2+1):end]
    hamiltonian(x,y,Γ)
end