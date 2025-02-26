import csdl_alpha as csdl


class BSplineParameterizationExplicitOperation(csdl.CustomExplicitOperation):
    """CSDL's CustomExplicitOperation for evaluating B-spline parameterization.
    """
    def __init__(self, num_radial, num_cp, jac):
        self.num_radial = num_radial
        self.num_cp = num_cp
        self.jac = jac

        super().__init__()

    def evaluate(self, control_points : csdl.Variable):
        num_cp = self.num_cp
        num_radial = self.num_radial
        if control_points.shape != (num_cp, ):
            raise Exception(f"Shape mismatch. The shape of control_points is {control_points.shape} but expected {(num_cp, )}")
        
        self.declare_input("control_points", control_points)

        radial_profile = self.create_output("radial_profile", (num_radial, ))

        return radial_profile

    def compute(self, input_vals, output_vals):
        jac = self.jac
        control_points = input_vals["control_points"]

        output_vals["radial_profile"] = jac @ control_points

    def compute_derivatives(self, inputs, outputs, derivatives):
        derivatives["radial_profile", "control_points"] = self.jac.toarray()
