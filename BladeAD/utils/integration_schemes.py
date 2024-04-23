import csdl_alpha as csdl


def integrate_quantity(quantity: csdl.Variable, scheme='trapezoidal'):
    acceptable_schemes = ['trapezoidal', 'Riemann', 'Simpson']
    if scheme not in acceptable_schemes:
        raise Exception(f"unknown integration scheme; implemented schemes are {acceptable_schemes}")
    shape = quantity.shape
    num_nodes = shape[0]
    num_radial = shape[1]
    num_azimuthal = shape[2]

    if scheme == 'Simpson' and (num_radial % 2 == 0):
        raise ValueError("If integration scheme is 'Simpson', 'num_radial' must be odd")

    qt_container = csdl.Variable(shape=(num_nodes, ), value=0)

    for i in csdl.frange(num_nodes):
        if scheme == 'trapezoidal':
            integrated_quantity = csdl.sum((quantity[i, 0, :] + quantity[i, -1, :]) / 2) \
                 + csdl.sum(quantity[i, 1:-1, :]) 
            qt_container = qt_container.set(
                csdl.slice[i], value=integrated_quantity / num_azimuthal
            )
        elif scheme == 'Simpson':
            integrated_quantity = csdl.sum(quantity[i, 0, :]) + csdl.sum(quantity[i, -1, :]) \
                + 4 * csdl.sum(quantity[i, 1:-1:2, :]) + 2 * csdl.sum(quantity[i, 2:-1:2, :])
            qt_container = qt_container.set(
                csdl.slice[i], value=integrated_quantity / 3 / num_azimuthal
            )
        elif scheme == 'Riemann':
            integrated_quantity = csdl.sum(quantity[i, :, :])
            qt_container = qt_container.set(
                csdl.slice[i], valu=integrated_quantity / num_azimuthal
            )
        else:
            raise NotImplementedError

    return qt_container


