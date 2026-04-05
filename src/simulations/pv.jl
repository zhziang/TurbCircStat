using Random, HDF5, ArgParse, ProgressMeter
import OrdinaryDiffEq as ODE
import DiffEqCallbacks as CB, NonlinearSolve as NLS
import SciMLBase as SMLB
include("pv_utils.jl")

#Parse argument
aps = ArgParseSettings()
@add_arg_table! aps begin
	"--npoints", "-n"
	help = "Number of point vortices (even number, half positive halfnegative)."
	arg_type = Int
	default = 10
	"--tspan"
	help = "Time span of the simulation"
	arg_type = Float64
	default = 100.0
	"--dt"
	help = "Time step size."
	arg_type = Float64
	default = 0.01
	"--href"
	help = "Reference Hamiltonian"
	arg_type = Float64
	default = 0.0
	"--path", "-p"
	help = "The path of the output file."
	arg_type = String
	default = @__DIR__
end
args = parse_args(aps)
npoints = args["npoints"]
tspan = args["tspan"]
dt = args["dt"]
hamiltonian_ref = args["href"]


Γ = 2 .* [ones(npoints÷2); -ones(npoints÷2)] ./ npoints

u₀ = rand(2 * npoints)

initial_hamiltonian = begin
	x, y = u₀[1:(end÷2)], u₀[(end÷2+1):end]
	hamiltonian(x, y, Γ)
end

function odefunc!(du, u, p, t)
	x, y = u[1:(end÷2)], u[(end÷2+1):end]
	du .= vec(vcat(velocity(x, y, Γ)...))
	return nothing
end

function isoHamiltonian_manifold(residual, u, p, t)
	x, y = u[1:(end÷2)], u[(end÷2+1):end]
	offset = (t ≤ tspan/10) ? 1-10t/tspan : 0
	residual .= (initial_hamiltonian-hamiltonian_ref)*offset + hamiltonian_ref - hamiltonian(x, y, Γ)
	return nothing
end

mproj = CB.ManifoldProjection(isoHamiltonian_manifold, autodiff = NLS.AutoForwardDiff(), resid_prototype = zeros(1))

prob = ODE.ODEProblem(odefunc!, u₀, (0, tspan))

sol = ODE.solve(prob, ODE.Vern7(); dt = dt, adaptive = false, save_everystep = false, callback = mproj)

H = map(sol.u) do u
	x, y = u[1:(end÷2)], u[(end÷2+1):end]
	hamiltonian(x, y, Γ)
end

results = hcat(sol.u[sol.t .≥ tspan/10]...)

if SMLB.successful_retcode(sol)
	output_path = args["path"] * "/.output/N$(npoints)H$(hamiltonian_ref).h5"
	isdir(args["path"] * "/.output/") || mkdir(args["path"] * "/.output/")
	h5open(output_path, "w") do fid
		write_dataset(fid, "pv Positions", results)
		write_dataset(fid, "ΔH", H[sol.t .≥ tspan/10] .- hamiltonian_ref)
	end
else
	error(sol.retcode)
end


