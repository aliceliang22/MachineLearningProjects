import numpy as np
# Linear regression find the best values of a and to minimize the square of errors
# (the difference between measured y and value calculated by the following linear equation):
#
#     y = a + b*x
#
#     y1 = a + b * x1
#     y2 = a + b * x2
#     ...
#     yn = a + b * xn
#
# a and b can be found by the following formula:
# (see https://towardsdatascience.com/linear-regression-derivation-d362ea3884c2
#
#   xa = (x1 + x2 + ... + xn) / n
#   ya = (y1 + y2 + ... + yn) / n
#   numerator = (x1*y1 + x2*y2 + ... + xn*yn) - ya * (x1 + x2 + ... + xn)
#   denominator = (x1*x1 + x2*x2 + ... + xn*xn) - xa * (x1 + x2 + ... + xn)
#   b = numerator / denominator
#   a = ya - b * xa
#
# Returns linear equation coeffients a and b
def get_linear_regression_with_list(x, y):

    nx = len(x)
    ny = len(y)

    sumx = 0
    sumy = 0
    sumxy = 0
    sumxx = 0
    for i in range(0, nx):
        sumx = sumx + x[i]
        sumy = sumy + y[i]
        sumxy = sumxy + x[i] * y[i]
        sumxx = sumxx + x[i] * x[i]

    xa = sumx / nx
    ya = sumy / ny
    numerator = sumxy - ya * sumx
    denominator = sumxx - xa * sumx
    b = numerator / denominator
    a = ya - b * xa

    return (a, b)

def get_linear_regression_with_numpy_array(x, y):

    xa = np.mean(x)
    ya = np.mean(y)
    sumx = np.sum(x)
    sumxy = np.sum(x*y)
    sumxx = np.sum(x*x)

    numerator = sumxy - ya * sumx
    denominator = sumxx - xa * sumx
    b = numerator / denominator
    a = ya - b * xa

    return (a, b)

# Return a list of y and a list of x
def get_regression_data_in_list():

    x = [53, 58, 93, 88, 39, 61, 24, 26, 87, 68, 9, 88, 0, 45, 60,
         15, 31, 96, 38, 7, 20, 81, 99, 48, 16, 30, 78, 41, 26, 65, 69]
    y = [59.03, 71.60, 100.57, 133.98, 50.13, 53.81, 26.37, 25.93, 135.50, 73.80, 0.39, 94.41, 22.85, 49.24, 82.00,
         32.10, 65.65, 99.21, 50.84, 24.96, 14.05, 80.69, 125.52, 52.30, 17.36, 55.62, 88.69, 52.65, 42.82, 84.60, 67.18]
    return (x, y)

# Return a numpy array of y and a numpy array of x
def get_regression_data_in_numpy_array():
    x = np.array([53, 58, 93, 88, 39, 61, 24, 26, 87, 68, 9, 88, 0, 45, 60,
         15, 31, 96, 38, 7, 20, 81, 99, 48, 16, 30, 78, 41, 26, 65, 69])
    y = np.array([59.03, 71.60, 100.57, 133.98, 50.13, 53.81, 26.37, 25.93, 135.50, 73.80, 0.39, 94.41, 22.85, 49.24, 82.00,
         32.10, 65.65, 99.21, 50.84, 24.96, 14.05, 80.69, 125.52, 52.30, 17.36, 55.62, 88.69, 52.65, 42.82, 84.60, 67.18])
    return (x, y)

# Test linear regression implementation using different methods
def test_regression_in_python():
    # Test linear regression implementation using list
    x_list, y_list = get_regression_data_in_list()
    a, b = get_linear_regression_with_list(x_list, y_list)
    print("From List:\t a = " + str(a) + "\t b = " + str(b))

    # Test linear regression implementation using numpy array
    x_array, y_array = get_regression_data_in_numpy_array()
    a, b = get_linear_regression_with_numpy_array(x_array, y_array)
    print("From Numpy:\t a = " + str(a) + "\t b = " + str(b))

def main():
    test_regression_in_python()

if __name__ == "__main__":
    main()
