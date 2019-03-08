import h5py

# run_name = 'dropout_MI_uncertainty'
run_name = 'dropconnect_MI_uncertainty'

h5f = h5py.File(run_name + '.h5', 'r')
y = h5f['y'][:]
y_pred = h5f['y_pred'][:]
y_var = h5f['y_var'][:]
h5f.close()

print()

