from sympy import symbols
from sympy.logic.boolalg import Or, And, simplify_logic

# Define Boolean variables
radius, texture, perimeter, area, smoothness, compactness, concavity, concave_points, symmetry, fractal_dimension = symbols(
    'radius texture perimeter area smoothness compactness concavity concave_points symmetry fractal_dimension'
)

# Construct the expression
expr = And(
        Or(
            Or(
                Or(
                    Or(
                        Or(
                            Or(
                                And(
                                    Or(radius, texture),
                                    perimeter
                                ),
                                area
                            ),
                            area
                        ),
                        compactness
                    ),
                    concavity
                ),
                concave_points
            ),
            symmetry
        ),
        fractal_dimension
    )


# Simplify it
simplified = simplify_logic(expr, form='dnf')  # Or 'cnf' if you prefer conjunctive form

print(simplified)
