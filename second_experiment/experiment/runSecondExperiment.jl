using StochasticPrograms
using JuMP
using GLPK
using Ipopt
using Distributed
using Distributions
using LinearAlgebra
using Logging
using Test
using Gurobi
using Random
using JLD2
using DelimitedFiles

@show(pathof(StochasticPrograms))
@show(pwd())

import StochasticPrograms: probability, expected

θ_plot_history = zeros(1)
Prob = readdlm(string(pwd(),"/list_problems.txt"))

stepsize_policy_num = 0;
stepsize_length_num = 0;

seed_num = 5; # the users can set any random seed number they want
prob_num = 1; # there are seven problems: {1,2,3,4,5,6,7} are corresponding to {SH31, SH10, 20term, gbd, LandS, ssn, storm}
stepsize_policy_num = 1; # there are two policies: {1,3} are corresponding to {constant, practical} step-size rules
stepsize_length_num = 4; # there are four step-size prameters: {1,2,3,4} are corresponding to 1/\rho = {10, 1, 0.1, 0.01} choices
scenario_num = 100; # only a choice of the number of scenarios is considered: |S_k| = 100

if stepsize_policy_num == 1
    stepsize_policy = "const"
    stepsize_policy_value = 1.0
else
    stepsize_policy = "practical"
    stepsize_policy_value = 3.0
end

if stepsize_length_num == 1
    stepsize_length_name = "10"
    stepsize_value = (0.1)^(-1)
elseif stepsize_length_num == 2
    stepsize_length_name = "1"
    stepsize_value = (0.1)^(0)
elseif stepsize_length_num == 3
    stepsize_length_name = "01"
    stepsize_value = (0.1)^(1)
else
    stepsize_length_name = "001"
    stepsize_value = (0.1)^(2)
end

include("../test/decisions/decisions.jl")

# Include suitable problems
prob_name = Prob[prob_num:prob_num, :]
if cmp(prob_name[1], "SH31") == 0
    include("problems/SH31_100.jl")
elseif cmp(prob_name[1], "SH10") == 0
    include("problems/SH10_100.jl")
else
    pathtofile = string(pwd(),"/experiment/smps/",prob_name[1],"/",prob_name[1],".smps")
    many_scenarios = false
    if many_scenarios
        global simple_smps = read(pathtofile, StochasticProgram, optimizer = LShaped.Optimizer)
    else
        global simple_smps = read(pathtofile, StochasticProgram, num_scenarios = scenario_num, optimizer = LShaped.Optimizer)
    end

    # Calculate the number of scenarios
    scenario_info = simple_smps.structure.scenarioproblems[1]
    N = length(scenario_info.scenarios)
end

# Set random seed
Random.seed!(seed_num)

# Set solvers
const GRB_ENV = Gurobi.Env()
subsolver = () -> Gurobi.Optimizer(GRB_ENV)
qpsolver = () -> begin
    opt = Ipopt.Optimizer()
    return opt
end

regularizers = [RegularizedDecomposition(penaltyterm = Quadratic())]

aggregators = [PartialAggregate(scenario_num)]

consolidators = [Consolidate()]

x_values_all = Vector{Vector{Float64}}()
x_outer_values_all = Vector{Vector{Float64}}()

if cmp(prob_name[1], "SH31") == 0 || cmp(prob_name[1], "SH10") == 0
    for (model,scenarios,name) in problems
        tol = 1e-5
        sp = instantiate(model, scenarios, optimizer = LShaped.Optimizer)
        set_silent(sp)
        for regularizer in regularizers, aggregator in aggregators, consolidator in consolidators
            set_optimizer_attribute(sp, Regularizer(), regularizer)
            set_optimizer_attribute(sp, Aggregator(), aggregator)
            set_optimizer_attribute(sp, Consolidator(), consolidator)
            set_optimizer_attribute(sp, MasterOptimizer(), subsolver)
            set_optimizer_attribute(sp, SubProblemOptimizer(), subsolver)

            # set step size
            regularizer.parameters.σ = stepsize_value
            regularizer.parameters.policy = stepsize_policy_value

            x_values_new, θ_plot = optimize!(sp)
            x_value = x_values_new[size(x_values_new,1)]
            append!(x_values_all, x_values_new)
            push!(x_outer_values_all, x_value)
            append!(θ_plot_history, θ_plot[2:end])
            append!(θ_plot_history, θ_plot[end])

            while size(x_values_all,1) < 10000

                # set step size
                regularizer.parameters.σ = stepsize_value
                regularizer.parameters.policy = stepsize_policy_value
                regularizer.parameters.θ_est = θ_plot_history[end]

                x_values_new, θ_plot = optimize!(sp, x_value)
                x_value = x_values_new[size(x_values_new,1)]
                append!(x_values_all, x_values_new)
                push!(x_outer_values_all, x_value)

                append!(θ_plot_history, θ_plot[2:end])
                append!(θ_plot_history, θ_plot[end])

            end
        end
    end
else
    tol = 1e-5
    for regularizer in regularizers, aggregator in aggregators, consolidator in consolidators

        set_optimizer_attribute(simple_smps, Regularizer(), regularizer)
        set_optimizer_attribute(simple_smps, Aggregator(), aggregator)
        set_optimizer_attribute(simple_smps, Consolidator(), consolidator)
        set_optimizer_attribute(simple_smps, MasterOptimizer(), subsolver)
        set_optimizer_attribute(simple_smps, SubProblemOptimizer(), subsolver)

        set_silent(simple_smps)

        # set step size
        regularizer.parameters.σ = stepsize_value
        regularizer.parameters.policy = stepsize_policy_value

        x_values_new, θ_plot = optimize!(simple_smps)
        x_value = x_values_new[size(x_values_new,1)]
        append!(x_values_all, x_values_new)
        push!(x_outer_values_all, x_value)

        append!(θ_plot_history, θ_plot[2:end])
        append!(θ_plot_history, θ_plot[end])

        while size(x_values_all,1) < 10000

            # set step size
            regularizer.parameters.σ = stepsize_value
            regularizer.parameters.policy = stepsize_policy_value
            regularizer.parameters.θ_est = θ_plot_history[end]

            global simple_smps = read(pathtofile, StochasticProgram, num_scenarios = scenario_num, optimizer = LShaped.Optimizer)

            set_optimizer_attribute(simple_smps, Regularizer(), regularizer)
            set_optimizer_attribute(simple_smps, Aggregator(), aggregator)
            set_optimizer_attribute(simple_smps, Consolidator(), consolidator)
            set_optimizer_attribute(simple_smps, MasterOptimizer(), subsolver)
            set_optimizer_attribute(simple_smps, SubProblemOptimizer(), subsolver)

            set_silent(simple_smps)

            x_values_new, θ_plot = optimize!(simple_smps, x_value)
            x_value = x_values_new[size(x_values_new,1)]
            append!(x_values_all, x_values_new)
            push!(x_outer_values_all, x_value)
            append!(θ_plot_history, θ_plot[2:end])
            append!(θ_plot_history, θ_plot[end])

        end
    end
end

if cmp(prob_name[1], "SH31") == 0 || cmp(prob_name[1], "SH10") == 0
    save_document = string(pwd(),"/data_",prob_name[1],"_",string(N),".jld2")
else
    save_document = string(pwd(),"/data_",prob_name[1],".jld2")
end
jldsave(save_document; x_info = x_values_all, x_outer_info = x_outer_values_all, θ_plot = θ_plot_history)




