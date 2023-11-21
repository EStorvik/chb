from dolfin import Expression


CrossInitialCondition = Expression(
    (
        "((x[0]<(0.5+delta) && x[0]>(0.5-delta)) && (x[1]<(0.5+2*delta) && x[1]>(0.5-2*delta))) || ((x[1]<(0.5+delta) && x[1]>(0.5-delta)) && (x[0]<(0.5+2*delta) && x[0]>(0.5-2*delta))) ? 1.0 : 0.0",
        "0.0",
    ),
    degree=0,
    delta=0.15,
)
