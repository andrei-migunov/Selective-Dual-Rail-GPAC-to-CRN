# Selective-Dual-Rail-GPAC-to-CRN
In Python, convert a GPAC to a CRN by dual-railing only the variables that require it.


basic_ops.selective_dual_rail converts a general-purpose analog computer (GPAC) into a chemical reaction network (CRN). Previous dual-railing methods either convert all the variables of the input system (as result, doubling the system size), or rely on *fast annihilation reactions*, which are not yet shown to be bounded in general.

This approach constructs a graph representing the ability of an ill-formed (non-CRN-implementable) variable to "infect" another variable in the system. Some variables are safe, and do not need to be dual-railed. In this sense, selective_dual_rail is selective. 

Please see our paper appearing in the proceedings of Unconventional Compoutation and Natural Computation 2025 for more details. The method emphasizes Tarjan's algorithm for finding strongly connected components the "can-infect" graph.: https://webusers.i3s.univ-cotedazur.fr/UCNC2025/accepted/#short

If you would like to run an example from within Python, try `tests.py` and please take a look at `example_input.txt` for input formatting guidelines. You can run the conversion from command line using

`py main.py "example_input.txt"`

In this example, two variables are dual-railed (including the leader variable that tracks the desired real-number computation). Since the leader variable is dual-railed, extra variable have to be introduced to re-create the value: these are `z` and `z_h'. `z_h' will be the new system's (the CRN's) leader variable. Two variables in the original system are not only CRN-implementable already, but are not affected by the dual-railing of other variables: these are `x_3` and `x_4`. Although dual-rail representations are substituted into the expression for `x_3`, this does not result in its becoming ill-formed. On the other hand, both `x_1` and `x_2` are dual-railed as x_1 = u_x_1 - v_x_1 and x_2 = u_x_2 - v_x_2.

The output for the example looks something like this (you can use the `clean_names` function in `basic_ops.py` to clean this up, somewhat, if you'd like to.:

`z_h: -z*z_h + 1
z: -u_x_1*z + v_x_1*z + 1
u_x_1: -u_x_1**2*u_x_2*v_x_1 - u_x_1**2*v_x_1*v_x_2 - u_x_1*u_x_2*v_x_1**2 - u_x_1*u_x_2*v_x_1 - u_x_1*v_x_1**2*v_x_2 - u_x_1*v_x_1*v_x_2 - u_x_1*v_x_1*x_4 + u_x_1*v_x_2 + u_x_2*v_x_1 + u_x_2
v_x_1: -u_x_1**2*u_x_2*v_x_1 - u_x_1**2*v_x_1*v_x_2 - u_x_1*u_x_2*v_x_1**2 - u_x_1*u_x_2*v_x_1 + u_x_1*u_x_2 - u_x_1*v_x_1**2*v_x_2 - u_x_1*v_x_1*v_x_2 - u_x_1*v_x_1*x_4 + v_x_1*v_x_2 + v_x_2 + x_4
u_x_2: -u_x_1*u_x_2**3*v_x_2 - 2*u_x_1*u_x_2**2*v_x_2**2 - u_x_1*u_x_2*v_x_2**3 + u_x_1*u_x_2*v_x_2 + u_x_1 - u_x_2**3*v_x_1*v_x_2 - 2*u_x_2**2*v_x_1*v_x_2**2 + u_x_2**2*v_x_1 - u_x_2*v_x_1*v_x_2**3 - u_x_2*v_x_1*v_x_2 - u_x_2*v_x_2*x_3 + v_x_1*v_x_2**2 + x_3
v_x_2: -u_x_1*u_x_2**3*v_x_2 - 2*u_x_1*u_x_2**2*v_x_2**2 + u_x_1*u_x_2**2 - u_x_1*u_x_2*v_x_2**3 - u_x_1*u_x_2*v_x_2 + u_x_1*v_x_2**2 - u_x_2**3*v_x_1*v_x_2 - 2*u_x_2**2*v_x_1*v_x_2**2 - u_x_2*v_x_1*v_x_2**3 + u_x_2*v_x_1*v_x_2 - u_x_2*v_x_2*x_3 + v_x_1
x_3: -u_x_2*x_3**2 + u_x_2*x_3 + v_x_2*x_3**2 - v_x_2*x_3 + 0.1*x_4
x_4: -x_3*x_4 + x_3`
