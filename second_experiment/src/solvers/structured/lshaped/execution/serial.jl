# MIT License
#
# Copyright (c) 2018 Martin Biel
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

using Random

"""
    SerialExecution

Functor object for using serial execution in a lshaped algorithm. Create by supplying a [`Serial`](@ref) object through `execution` in the `LShapedSolver` factory function and then pass to a `StochasticPrograms.jl` model.

"""
struct SerialExecution{T <: AbstractFloat,
                       A <: AbstractVector,
                       F <: AbstractFeasibilityAlgorithm,
                       I <: AbstractIntegerAlgorithm} <: AbstractLShapedExecution
    subproblems::Vector{SubProblem{T,F,I}}
    decisions::DecisionMap
    subobjectives::A
    model_objectives::A
    metadata::MetaData

    function SerialExecution(structure::StageDecompositionStructure{2, 1, <:Tuple{ScenarioProblems}},
                             feasibility_strategy::AbstractFeasibilityStrategy,
                             integer_strategy::AbstractIntegerStrategy,
                             ::Type{T},
                             ::Type{A}) where {T <: AbstractFloat,
                                               A <: AbstractVector}
        F = worker_type(feasibility_strategy)
        I = worker_type(integer_strategy)
        execution = new{T,A,F,I}(Vector{SubProblem{T,F,I}}(), structure.decisions[2], A(), A(), MetaData())
        # Load subproblems
        for i in 1:num_subproblems(structure, 2)
            push!(execution.subproblems, SubProblem(
                subproblem(structure, 2, i),
                i,
                T(probability(scenario(structure, 2, i))),
                feasibility_strategy,
                integer_strategy))
        end
        return execution
    end
end

function num_thetas(lshaped::AbstractLShaped, ::SerialExecution)
    return num_thetas(num_subproblems(lshaped), lshaped.aggregation)
end

function initialize_subproblems!(execution::SerialExecution{T},
                                 scenarioproblems::ScenarioProblems,
                                 feasibility_strategy::AbstractFeasibilityStrategy,
                                 integer_strategy::AbstractIntegerStrategy) where T <: AbstractFloat
    for i in 1:num_subproblems(scenarioproblems)
        push!(execution.subproblems, SubProblem(
            subproblem(scenarioproblems, i),
            i,
            T(probability(scenario(scenarioproblems, i))),
            feasibility_strategy,
            integer_strategy))
    end
    return nothing
end

function mutate_subproblems!(mutator::Function, execution::SerialExecution)
    for subproblem in execution.subproblems
        mutator(subproblem)
    end
    return nothing
end

function finish_initilization!(lshaped::AbstractLShaped, execution::SerialExecution)
    append!(execution.subobjectives, fill(1e10, num_thetas(lshaped)))
    append!(execution.model_objectives, fill(-1e10, num_thetas(lshaped)))
    return nothing
end

function restore_subproblems!(lshaped::AbstractLShaped, execution::SerialExecution)
    for subproblem in execution.subproblems
        restore_subproblem!(subproblem)
    end
    return nothing
end

function evaluate_inexact_func_add_cut!(lshaped::AbstractLShaped, execution::SerialExecution{T}) where T <: AbstractFloat

    # Update subproblems
    update_known_decisions!(execution.decisions, lshaped.x)

    # The number of scenarios to generate an inexact cut
    num_scen_inexact = lshaped.aggregation.size

    if lshaped.data.consider_initial_scen != true
        lshaped.data.vec_scen = Random.shuffle(Vector(1:lshaped.num_subproblems))[1:num_scen_inexact]
        vec_scen = lshaped.data.vec_scen
        lshaped.data.scen_prob = sum(execution.subproblems[index].probability for index in lshaped.data.vec_scen)
    else
        vec_scen = lshaped.data.vec_scen
    end

    # Update and solve subproblems
    for index in lshaped.data.vec_scen
        subproblem = execution.subproblems[index]
        update_subproblem!(subproblem)
        cut::SparseHyperPlane{T} = subproblem(lshaped.x, lshaped.data.scen_prob, execution.metadata)
        aggregate_cut!(lshaped, lshaped.aggregation, cut)
    end

    return current_objective_value(lshaped) 

end

function evaluate_inexact_func!(lshaped::AbstractLShaped, execution::SerialExecution{T}) where T <: AbstractFloat

    # Update subproblems
    update_known_decisions!(execution.decisions, lshaped.x)

    # The number of scenarios to generate an inexact cut
    vec_scen = lshaped.data.vec_scen

    # Update and solve subproblems
    for index in vec_scen
        subproblem = execution.subproblems[index]
        update_subproblem!(subproblem)
        cut::SparseHyperPlane{T} = subproblem(lshaped.x, lshaped.data.scen_prob, execution.metadata)
        aggregate_cut_only!(lshaped, lshaped.aggregation, cut)
    end

    return current_objective_value_inner(lshaped) 

end

function evaluate_exact_func!(lshaped::AbstractLShaped, execution::SerialExecution{T}) where T <: AbstractFloat

    # Update subproblems
    update_known_decisions!(execution.decisions, lshaped.x)

    # The number of scenarios to generate an inexact cut
    vec_scen = Vector(1:lshaped.num_subproblems)

    # Update and solve subproblems
    for index in vec_scen
        subproblem = execution.subproblems[index]
        update_subproblem!(subproblem)
        cut::SparseHyperPlane{T} = subproblem(lshaped.x, lshaped.data.scen_prob, execution.metadata)
        aggregate_cut_only_exact!(lshaped, lshaped.aggregation, cut) 
    end

    return current_objective_value_inner(lshaped) 

end

function add_single_cut!(lshaped::AbstractLShaped, execution::SerialExecution{T}, δQ::AbstractVector, q::Float64) where T <: AbstractFloat

    # Update subproblems
    update_known_decisions!(execution.decisions, lshaped.x)

    vec_scen = lshaped.data.vec_scen

    # Update and solve subproblems
    index = vec_scen[1]
    subproblem = execution.subproblems[index]
    update_subproblem!(subproblem)
    cut::SparseHyperPlane{T} = subproblem(lshaped.x, subproblem.probability, execution.metadata, δQ, q)
    aggregate_single_cut!(lshaped, lshaped.aggregation, cut)

    return current_objective_value(lshaped) 

end

function resolve_subproblems!(lshaped::AbstractLShaped, execution::SerialExecution{T}) where T <: AbstractFloat

    # Update subproblems
    update_known_decisions!(execution.decisions, lshaped.x)
    # Assume no cuts are added
    added = false

    # The number of scenarios to generate an inexact cut
    num_scen_inexact = lshaped.aggregation.size

    vec_scen_all = Random.shuffle(Vector(1:lshaped.num_subproblems))
    vec_scen = vec_scen_all[1:num_scen_inexact]
    counter = 0
    lshaped.data.scen_prob = 0.0

    # Update and solve subproblems
    for subproblem in execution.subproblems
        counter = counter + 1
        if counter ∉ vec_scen
            continue
        end
        lshaped.data.scen_prob = lshaped.data.scen_prob + subproblem.probability
        update_subproblem!(subproblem)
        cut::SparseHyperPlane{T} = subproblem(lshaped.x, execution.metadata)
        added |= aggregate_cut!(lshaped, lshaped.aggregation, cut) 
    end

    added |= flush!(lshaped, lshaped.aggregation)

    # Return current objective value and cut_added flag
    return current_objective_value(lshaped), added
end

# API
# ------------------------------------------------------------
function (execution::Serial)(structure::StageDecompositionStructure{2, 1, <:Tuple{ScenarioProblems}},
                             feasibility_strategy::AbstractFeasibilityStrategy,
                             integer_strategy::AbstractIntegerStrategy,
                             ::Type{T},
                             ::Type{A}) where {T <: AbstractFloat,
                                               A <: AbstractVector}
    return SerialExecution(structure,
                           feasibility_strategy,
                           integer_strategy,
                           T,
                           A)
end

function str(::Serial)
    return ""
end
