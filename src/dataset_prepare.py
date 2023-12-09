from importer import import_library

# import numpy,pandas,matplotlib,scipy,os using import_library
np = import_library("numpy")
pd = import_library("pandas")
plt = import_library("matplotlib.pyplot")
signal = import_library("scipy.signal")
os = import_library("os")
sg = signal.signal

# import interpid  whit import_library 

from scipy.interpolate import interp1d


def dataset_prepare(input_gt_name, input_rpm_name):
    # Read the data from the CSV file
    data = pd.read_csv(input_gt_name)
    data_rpm = pd.read_csv(input_rpm_name)
    
    # Extract the x, y, and z positions
    x = data['x'].to_numpy()
    y = data['y'].to_numpy()
    z = data['z'].to_numpy()
    
    # Convert timestamps from Unix epoch to microseconds
    timestamps = data['timestamps'].to_numpy()
    
    # Extract the rpms
    rpms = data_rpm[['rosbagTimestamp', 'rpm1', 'rpm2', 'rpm3', 'rpm4']].to_numpy()
    timestamps_rpm = rpms[:, 0]
    timestamps_rpm = timestamps_rpm/1000
    rpms = rpms[:, 1:]
    rpm1 = rpms[:, 0]
    rpm2 = rpms[:, 1]
    rpm3 = rpms[:, 2]
    rpm4 = rpms[:, 3]
    
    # Interpolate the ground truth data to be at exactly 360 Hz
    t_curr = timestamps[0]
    dt = 1000000/360
    new_times_gt = [t_curr]
    while t_curr < timestamps[-1] - dt - 0.001:
        t_curr = t_curr + dt
        new_times_gt.append(t_curr)
    new_times_gt = np.asarray(new_times_gt)
    x = interp1d(timestamps[:], x[:], axis=0)(new_times_gt)
    y = interp1d(timestamps[:], y[:], axis=0)(new_times_gt)
    z = interp1d(timestamps[:], z[:], axis=0)(new_times_gt)
    time_deltas = np.diff(new_times_gt/1000000)
    timestamps = new_times_gt
    
    # Filter the data with a low pass butterworth filter at 5 Hz
    freq = 5
    b, a = sg.butter(1, freq/180, 'low', analog=False)
    butterx = sg.filtfilt(b, a, x)
    buttery = sg.filtfilt(b, a, y)
    butterz = sg.filtfilt(b, a, z)
    
    # Calculate the velocity
    vx = np.diff(butterx) / time_deltas
    vy = np.diff(buttery) / time_deltas
    vz = np.diff(butterz) / time_deltas
    
    # Filter the data with a 4th order low pass butterworth filter at 1 Hz
    freq = 1
    b, a = sg.butter(4, freq/180, 'low', analog=False)
    buttervx = sg.filtfilt(b, a, vx)
    buttervy = sg.filtfilt(b, a, vy)
    buttervz = sg.filtfilt(b, a, vz)
    
    # Calculate the acceleration
    ax = np.diff(buttervx) / time_deltas[1:]
    ay = np.diff(buttervy) / time_deltas[1:]
    az = np.diff(buttervz) / time_deltas[1:]
    
    # We down sample to RPM rate

    # get initial and final times for interpolations
    idx_s = 0
    for ts in timestamps_rpm:
        if ts > timestamps[0]:
            break
        else:
            idx_s = idx_s + 1
    assert idx_s < len(timestamps_rpm)
    idx_e = len(timestamps_rpm) - 1
    for ts in reversed(timestamps_rpm):
        if ts < timestamps[-1]:
            break
        else:
            idx_e = idx_e - 1
    assert idx_e > 0
    timestamps_rpm = timestamps_rpm[idx_s:idx_e + 1]
    rpm1 = rpm1[idx_s:idx_e + 1]
    rpm2 = rpm2[idx_s:idx_e + 1]
    rpm3 = rpm3[idx_s:idx_e + 1]
    rpm4 = rpm4[idx_s:idx_e + 1]

    # interpolate ground-truth samples at thrust times
    groundtruth_pos_data = interp1d(timestamps, np.array([x, y, z]).T, axis=0)(timestamps_rpm)
    groundtruth_vel_data = interp1d(timestamps[1:], np.array([vx, vy, vz]).T, axis=0)(timestamps_rpm)
    groundtruth_acc_data = interp1d(timestamps[2:], np.array([ax, ay, az]).T, axis=0)(timestamps_rpm)
    
    timestamps = timestamps_rpm
    
    # Print the acceleration data to a CSV file
    output_acc_data = pd.DataFrame()
    output_acc_data['timestamps'] = timestamps
    output_acc_data['ax'] = groundtruth_acc_data[:, 0]
    output_acc_data['ay'] = groundtruth_acc_data[:, 1]
    output_acc_data['az'] = groundtruth_acc_data[:, 2]
    
    filtered_gt_name = os.path.join(os.path.dirname(input_gt_name), 'groundTruth_acc')
    output_acc_data.to_csv(filtered_gt_name, index=False)
    
    # Print the rpm data to a CSV file
    output_rpm_data = pd.DataFrame()
    output_rpm_data['timestamps'] = timestamps_rpm
    output_rpm_data['rpm1'] = rpm1
    output_rpm_data['rpm2'] = rpm2
    output_rpm_data['rpm3'] = rpm3
    output_rpm_data['rpm4'] = rpm4
    
    rpm_file_name = os.path.join(os.path.dirname(input_rpm_name), 'groundTruth_rpm')
    output_rpm_data.to_csv(rpm_file_name, index=False)
    
    return filtered_gt_name, rpm_file_name
