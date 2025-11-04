using JuMP
using Ipopt

# === Sets ===
G = 1:9        # Generators
C = 1:7        # Consumers
N = 1:11       # Nodes
L = [
    (1,2),(1,11),(2,3),(2,11),(3,4),(3,9),
    (4,5),(5,6),(5,8),(6,7),(7,8),(7,9),
    (8,9),(9,10),(10,11)
]

# === Node mappings ===
gen_nodes = [2,2,2,3,4,5,7,9,9]
cons_nodes = [1,4,6,8,9,10,11]
Gk = Dict(k => findall(i -> gen_nodes[i] == k, G) for k in N)
Ck = Dict(k => findall(i -> cons_nodes[i] == k, C) for k in N)

# === Parameters ===
c  = Dict(i => [175,100,150,150,300,350,400,300,200][i] for i in G)
Pmax = Dict(i => [0.02,0.15,0.08,0.07,0.04,0.17,0.17,0.26,0.05][i] for i in G)
D  = Dict(j => [0.10,0.19,0.11,0.09,0.21,0.05,0.04][j] for j in C)
Qload = Dict(j => 0.0 for j in C)  # no reactive demand

# === Line parameters ===
g_vals = [4.12,5.67,2.41,2.78,1.98,3.23,1.59,1.71,1.26,1.11,1.32,2.01,4.41,2.14,5.06]
b_vals = [-20.1,-22.3,-16.8,-17.2,-11.7,-19.4,-10.8,-12.3,-9.2,-13.9,-8.7,-11.3,-7.7,-13.5,-26.7]
g = Dict(L[i] => g_vals[i] for i in 1:length(L))
b = Dict(L[i] => b_vals[i] for i in 1:length(L))

# === Model ===
model = Model(Ipopt.Optimizer)

# === Variables ===
@variable(model, 0 <= P[i in G] <= Pmax[i])                  # Active power gen
@variable(model, -0.03*Pmax[i] <= Q[i in G] <= 0.03*Pmax[i]) # Reactive power gen
@variable(model, 0.98 <= v[k in N] <= 1.02)                  # Voltage magnitude
@variable(model, -pi <= theta[k in N] <= pi)                 # Voltage angle

# Bidirectional flows: one variable per direction per line
@variable(model, p[e in L])      # flow k -> l
@variable(model, q[e in L])
@variable(model, p_rev[e in L])  # flow l -> k
@variable(model, q_rev[e in L])

# Reference angle
@constraint(model, theta[1] == 0.0)

# Initial guesses for better convergence
for k in N
    set_start_value(v[k], 1.0)
    set_start_value(theta[k], 0.0)
end

# === Objective ===
@objective(model, Min, sum(c[i]*P[i] for i in G))

# === Power balance constraints ===
for k in N
    # Active power balance
    @constraint(model,
        sum(P[i] for i in Gk[k]) - sum(D[j] for j in Ck[k]) ==
        sum(p[e] for e in L if e[1] == k) - sum(p[e] for e in L if e[2] == k) +
        sum(p_rev[e] for e in L if e[2] == k) - sum(p_rev[e] for e in L if e[1] == k)
    )

    # Reactive power balance
    @constraint(model,
        sum(Q[i] for i in Gk[k]) ==
        sum(q[e] for e in L if e[1] == k) - sum(q[e] for e in L if e[2] == k) +
        sum(q_rev[e] for e in L if e[2] == k) - sum(q_rev[e] for e in L if e[1] == k)
    )
end

# === AC Power flow equations (both directions) ===
for e in L
    k, l = e
    # Flow k -> l
    @NLconstraint(model,
        p[e] == v[k]^2 * g[e] -
                 v[k]*v[l]*(g[e]*cos(theta[k]-theta[l]) + b[e]*sin(theta[k]-theta[l]))
    )
    @NLconstraint(model,
        q[e] == -v[k]^2 * b[e] +
                 v[k]*v[l]*(b[e]*cos(theta[k]-theta[l]) - g[e]*sin(theta[k]-theta[l]))
    )

    # Flow l -> k
    @NLconstraint(model,
        p_rev[e] == v[l]^2 * g[e] -
                    v[k]*v[l]*(g[e]*cos(theta[l]-theta[k]) + b[e]*sin(theta[l]-theta[k]))
    )
    @NLconstraint(model,
        q_rev[e] == -v[l]^2 * b[e] +
                    v[k]*v[l]*(b[e]*cos(theta[l]-theta[k]) - g[e]*sin(theta[l]-theta[k]))
    )
end

# === Solve ===
optimize!(model)

# === Output ===
println("\n=== Optimization Results ===")
println("Objective (total cost): ", objective_value(model))
println("\n--- Generator Outputs ---")
for i in G
    println("Generator $i:  P = ", value(P[i]), "  Q = ", value(Q[i]))
end

println("\n--- Node Voltages ---")
for k in N
    println("Node $k:  V = ", value(v[k]), "  θ = ", value(theta[k]))
end
