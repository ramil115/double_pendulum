import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from double_pendulum.model.model_parameters import model_parameters
from double_pendulum.system_identification.dynamics import build_identification_matrices
from double_pendulum.system_identification.optimization import solve_least_squares
from double_pendulum.system_identification.plotting import plot_torques


def run_system_identification(measured_data_csv, fixed_mpar, variable_mpar, mp0, bounds):

    Q, phi = build_identification_matrices(fixed_mpar, variable_mpar, measured_data_csv)

    mp_opt = solve_least_squares(Q, phi, mp0, bounds)

    print('Identified Parameters:')
    for i in range(len(variable_mpar)):
        print("{:10s} = {:+.3e}".format(variable_mpar[i], mp_opt[i]))

    # calculate errors
    Q_opt = phi.dot(mp_opt)
    mae = mean_absolute_error(Q.flatten(), Q_opt.flatten())
    rmse = mean_squared_error(Q.flatten(), Q_opt.flatten(), squared=False)

    print("Mean absolute error: ", mae)
    print("Mean root mean squared error: ", rmse)

    # plotting results
    data = pd.read_csv(measured_data_csv)
    time = data["time"].tolist()
    plot_torques(time, Q[::2, 0], Q[1::2, 0], Q_opt[::2], Q_opt[1::2])

    all_par = fixed_mpar
    for i, key in enumerate(variable_mpar):
        if key == "m1r1":
            all_par["m1"] = mp_opt[i] / fixed_mpar["l1"]
            all_par["r1"] = fixed_mpar["l1"]
        elif key == "m2r2":
            all_par["r2"] = mp_opt[i] / mp_opt[i+1]
            # this requires the order ..., "m2r2", "m2", .. in variable_mpar
            # Todo: find better solution
        else:
            all_par[key] = mp_opt[i]
    mpar = model_parameters()
    mpar.load_dict(all_par)

    return mp_opt, mpar