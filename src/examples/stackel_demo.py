from triaxial_jeans.stackel import StackelPotential


def generator(t):
    """
    Example Stäckel function.

    Real models will use functions
    derived from the density profile.
    """
    return -t*t


model = StackelPotential(generator)

q = (10,2,0.2)

print("Phi =", model.potential(*q))
print("Force =", model.force(*q))
