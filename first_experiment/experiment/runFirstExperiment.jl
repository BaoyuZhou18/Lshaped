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

Q_plot_history = zeros(1)
θ_plot_history = zeros(1)
Q_curr_plot_history = zeros(1)
Q_true_plot_history = zeros(1)
Q_true_outer_plot_history = zeros(1)

seed_num = 5; # the users can set any random seed number they want
prob_num = 1; # only one problem (SH31) is tested here
stepsize_policy_num = 3; # there are three policies: {1,2,3} are corresponding to {constant, ideal, practical} step-size rules
stepsize_length_num = 5; # there are five step-size prameters: {1,2,3,4,5} are corresponding to 1/\rho = {10, 1, 0.1, 0.01, 0.001} choices
scenario_num = 100; # only a choice of the number of scenarios is considered: |S_k| = 100

if stepsize_policy_num == 1
    stepsize_policy = "const"
    stepsize_policy_value = 1.0
elseif stepsize_policy_num == 2
    stepsize_policy = "ideal"
    stepsize_policy_value = 2.0
else
    stepsize_policy = "practical"
    stepsize_policy_value = 3.0
end

if stepsize_length_num == 1
    stepsize_length_name = "10"
    stepsize_value = 10
elseif stepsize_length_num == 2
    stepsize_length_name = "1"
    stepsize_value = 1
elseif stepsize_length_num == 3
    stepsize_length_name = "01"
    stepsize_value = 0.1
elseif stepsize_length_num == 4
    stepsize_length_name = "001"
    stepsize_value = 0.01
elseif stepsize_length_num == 5
    stepsize_length_name = "0001"
    stepsize_value = 0.001
end

include("problems/SH31_100.jl")

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

        x_value, Q_plot, θ_plot, Q_curr_plot, Q_true_plot, Q_true_outer_plot = optimize!(sp)

        append!(Q_plot_history, Q_plot[2:end])
        append!(θ_plot_history, θ_plot[2:end])
        append!(θ_plot_history, θ_plot[end])
        append!(Q_curr_plot_history, Q_curr_plot[2:end])
        append!(Q_true_plot_history, Q_true_plot[2:end])
        append!(Q_true_outer_plot_history, Q_true_outer_plot[2:end])

        while size(Q_plot_history,1) < 1000

            # set step size
            regularizer.parameters.σ = stepsize_value
            regularizer.parameters.policy = stepsize_policy_value
            regularizer.parameters.θ_est = θ_plot_history[end]

            x_value, Q_plot, θ_plot, Q_curr_plot, Q_true_plot, Q_true_outer_plot = optimize!(sp, x_value)

            append!(Q_plot_history, Q_plot[2:end])
            append!(θ_plot_history, θ_plot[2:end])
            append!(θ_plot_history, θ_plot[end])
            append!(Q_curr_plot_history, Q_curr_plot[2:end])
            append!(Q_true_plot_history, Q_true_plot[2:end])
            append!(Q_true_outer_plot_history, Q_true_outer_plot[2:end])

        end
    end
end

# Save data
save_document = string(pwd(),"/",stepsize_policy,"_",stepsize_length_name,"_SH31.jld2")
jldsave(save_document; Q_plot = Q_plot_history, θ_plot = θ_plot_history, Q_curr_plot = Q_curr_plot_history, Q_true_plot = Q_true_plot_history, Q_true_outer_plot = Q_true_outer_plot_history)
