# Selective-Dual-Rail-GPAC-to-CRN
In Python, convert a GPAC to a CRN by dual-railing only the variables that require it.


basic_ops.selective_dual_rail converts a general-purpose analog computer (GPAC) into a chemical reaction network (CRN). Previous dual-railing methods either convert all the variables of the input system (as result, doubling the system size), and other methods rely on *fast annihilation reactions*, which are not yet shown to be bounded.

This approach constructs a graph representing the ability of ill-formed (non-CRN-implementable) variables to `infect` other variables in the system. Some variables are safe, and do not need to be dual-railed. In this sense, selective_dual_rail is selective. 

Please see our paper appearing in the proceedings of Unconventional Compoutation and Natural Computation 2025 : https://webusers.i3s.univ-cotedazur.fr/UCNC2025/accepted/#short

Please run tests.py for an example, and please take a look at `example_input.txt` for input formatting guidelines. You can run this from command line using

`py main.py "example_input.txt"`

In this example, two variables are dual-railed (including the leader variable that tracks the desired real-number computation). Since the leader variable is dual-railed, extra variable have to be introduced to re-create the value: these are `z` and `z_h'. `z_h' will be the new system's (the CRN's) leader variable. Two variables in the original system are not only CRN-implementable already, but are not affected by the dual-railing of other variables: these are `x_3` and `x_4'. Although dual-rail representations are substituted into the expression for `x_3', this does not result in its becoming ill-formed. On the other hand, both `x_1` and `x_2` are dual-railed as x_1 = u_x_1 - v_x_1 and x_2 = u_x_2 - v_x_2.
