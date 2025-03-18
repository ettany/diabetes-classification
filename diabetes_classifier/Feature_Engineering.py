def mean(values):
    # Calculate the mean of a dataset
    return sum(values) / len(values)

def variance(values):
    # Calculate the variance of a dataset
    avg = mean(values)
    return sum((x - avg) ** 2 for x in values) / len(values)

def standard_deviation(values):
    # Calculate the standard deviation of a dataset
    return variance(values) **  (1/2)
