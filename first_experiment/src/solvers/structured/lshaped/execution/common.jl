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

function num_thetas(lshaped::AbstractLShaped, ::AbstractLShapedExecution)
    return num_thetas(num_subproblems(lshaped),
                      lshaped.aggregation,
                      scenarioproblems(lshaped.structure))
end

function timestamp(lshaped::AbstractLShaped, ::AbstractLShapedExecution)
    return lshaped.data.inner_iter
end

function current_decision(lshaped::AbstractLShaped, ::AbstractLShapedExecution)
    return lshaped.x
end

function incumbent_decision(::AbstractLShaped, ::Integer, regularization::AbstractRegularization, ::AbstractLShapedExecution)
    return map(regularization.ξ) do ξᵢ
        return ξᵢ.value
    end
end

function incumbent_objective(::AbstractLShaped, ::Integer, regularization::AbstractRegularization, ::AbstractLShapedExecution)
    return regularization.data.Q̃
end

function incumbent_trustregion(::AbstractLShaped, ::Integer, rd::RegularizedDecomposition, ::AbstractLShapedExecution)
    return rd.data.σ
end

function incumbent_trustregion(::AbstractLShaped, ::Integer, tr::TrustRegion, ::AbstractLShapedExecution)
    Δ = StochasticPrograms.decision(tr.decisions, tr.data.Δ)
    return Δ.value
end

function start_workers!(::AbstractLShaped, ::AbstractLShapedExecution)
    return nothing
end

function close_workers!(::AbstractLShaped, ::AbstractLShapedExecution)
    return nothing
end

function readd_cuts!(lshaped::AbstractLShaped, consolidation::Consolidation, ::AbstractLShapedExecution)
    for i in eachindex(consolidation.cuts)
        for cut in consolidation.cuts[i]
            add_cut!(lshaped, cut; consider_consolidation = false, check = false)
        end
        for cut in consolidation.feasibility_cuts[i]
            add_cut!(lshaped, cut; consider_consolidation = false, check = false)
        end
    end
    return nothing
end

function subobjectives(lshaped::AbstractLShaped, execution::AbstractLShapedExecution)
    return execution.subobjectives
end

function set_subobjectives(lshaped::AbstractLShaped, Qs::AbstractVector, execution::AbstractLShapedExecution)
    execution.subobjectives .= Qs
    return nothing
end

function model_objectives(lshaped::AbstractLShaped, execution::AbstractLShapedExecution)
    return execution.model_objectives
end

function set_model_objectives(lshaped::AbstractLShaped, θs::AbstractVector, execution::AbstractLShapedExecution)
    ids = active_model_objectives(lshaped)
    execution.model_objectives[ids] .= θs[ids]
    return nothing
end

function solve_master!(lshaped::AbstractLShaped, ::AbstractLShapedExecution)
    try
        MOI.optimize!(lshaped.master)
    catch err
        status = MOI.get(lshaped.master, MOI.TerminationStatus())
        # Master problem could not be solved for some reason.
        @unpack Q,θ = lshaped.data
        gap = abs(θ-Q)/(abs(Q)+1e-10)
        # Always print this warning
        @warn "Master problem could not be solved, solver returned status $status. The following relative tolerance was reached: $(@sprintf("%.1e",gap)). Aborting procedure."
        rethrow(err)
    end
    return MOI.get(lshaped.master, MOI.TerminationStatus())
end

function iterate!(lshaped::AbstractLShaped, ::AbstractLShapedExecution)

    # Initialize inner loop's termination flag
    lshaped.data.inner_term_cond = false
    lshaped.data.consider_initial_scen = false

    # Initialize intermediate vector
    num_var = length(lshaped.x)
    s = Vector{Float64}(undef,num_var)

    # export coefficient for master problem
    master_coeff = Vector{Float64}(undef,num_var)
    for i=1:num_var
        master_coeff[i] = value(lshaped.data.master_objective.decision_part.terms[i].coefficient)
    end

    # evaluate initial inexact function
    Q = evaluate_inexact_func_add_cut!(lshaped)

    if Q == Inf 
        # Early termination log
        log!(lshaped; status = MOI.INFEASIBLE)
        return MOI.INFEASIBLE
    end
    if Q == -Inf
        # Early termination log
        log!(lshaped; status = MOI.DUAL_INFEASIBLE)
        return MOI.DUAL_INFEASIBLE
    end
    lshaped.data.Q = Q
    push!(lshaped.data.Q_plot, lshaped.data.Q)

    Q_true = evaluate_exact_func!(lshaped)
    push!(lshaped.data.Q_true_plot, Q_true)
    push!(lshaped.data.Q_true_outer_plot, Q_true)

    x_old = copy(lshaped.x)

    ratio = lshaped.regularization.data.σ

    # Updates for ideal and practical step sizes
    if lshaped.regularization.parameters.policy == 2.0
        lshaped.regularization.data.σ = ratio / (Q_true - 24.988121536932987)
        lshaped.regularization.parameters.σ = ratio / (Q_true - 24.988121536932987)
    elseif lshaped.regularization.parameters.policy == 3.0
        if lshaped.regularization.parameters.θ_est != 0.0
            if lshaped.data.Q > lshaped.regularization.parameters.θ_est
                lshaped.regularization.parameters.σ = ratio / (lshaped.data.Q - lshaped.regularization.parameters.θ_est)
                lshaped.regularization.data.σ = ratio / (lshaped.data.Q - lshaped.regularization.parameters.θ_est)
            end
	    end
    end
    
    # evaluate current inexact function
    Q_curr = evaluate_inexact_func!(lshaped)
    push!(lshaped.data.Q_curr_plot, Q_curr)

    # Solve master problem
    status = solve_master!(lshaped)
    if !(status ∈ AcceptableTermination)
        # Early termination log
        log!(lshaped; status = status)
        return status
    end

    # Update master solution
    update_solution!(lshaped)

    # evaluate current inexact function
    Q_curr = evaluate_inexact_func!(lshaped)

    lshaped.data.consider_initial_scen = true

    lshaped.data.θ = calculate_estimate(lshaped)

    while true

        push!(lshaped.data.Q_plot, lshaped.data.Q)
        push!(lshaped.data.θ_plot, lshaped.data.θ)
        push!(lshaped.data.Q_curr_plot, Q_curr)

	    Q_true = evaluate_exact_func!(lshaped)
        push!(lshaped.data.Q_true_plot, Q_true)

        x_old = copy(lshaped.x)

        if lshaped.data.beta * (lshaped.data.Q - lshaped.data.θ) <= lshaped.data.Q - Q_curr || lshaped.data.Q - lshaped.data.θ < 1e-6

            lshaped.data.inner_term_cond = true

            lshaped.regularization.data.Q = lshaped.data.Q
            lshaped.regularization.data.θ = lshaped.data.θ

            take_step!(lshaped)
            # Log progress
            log!(lshaped)
            # clear consolidation history
            consolidate_clear!(lshaped, lshaped.consolidation)

            global x_value = lshaped.x

            break;
        else
            # Log progress
            lshaped.data.inner_term_cond = false
            log!(lshaped)
        end

        consolidate!(lshaped, lshaped.consolidation)

        # Compute a new cut
        for i=1:num_var
            if i in eachindex(lshaped.regularization.ξ)
                s[i] = (lshaped.regularization.ξ[i].value - lshaped.x[i]) / lshaped.regularization.data.σ
            else
                s[i] = -lshaped.x[i] / lshaped.regularization.data.σ
            end
        end
        
        δQ_cand = master_coeff - s
        q_cand = lshaped.data.θ - sum(s[i]*lshaped.x[i] for i=1:num_var)

        # Add a single cut
        Q_cand = add_single_cut!(lshaped, δQ_cand, q_cand)

        # evaluate initial inexact function
        Q = evaluate_inexact_func_add_cut!(lshaped)

        if Q == Inf 
            # Early termination log
            log!(lshaped; status = MOI.INFEASIBLE)
            return MOI.INFEASIBLE
        end
        if Q == -Inf
            # Early termination log
            log!(lshaped; status = MOI.DUAL_INFEASIBLE)
            return MOI.DUAL_INFEASIBLE
        end

        status = solve_master!(lshaped)
        if !(status ∈ AcceptableTermination)
            # Early termination log
            log!(lshaped; status = status)
            return status
        end

        # Update master solution
        update_solution!(lshaped)

        # evaluate current inexact function
        Q_curr = evaluate_inexact_func!(lshaped)

        lshaped.data.consider_initial_scen = true

        lshaped.data.θ = calculate_estimate(lshaped)

    end

    return nothing
end
