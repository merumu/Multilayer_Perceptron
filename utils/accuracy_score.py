def accuracy_score_(y_true, y_pred, normalize=True):
    """
    Compute the accuracy score.
    Args:
        y_true: a scalar or a numpy ndarray for the correct labels
        y_pred: a scalar or a numpy ndarray for the predicted labels
    Returns:
        The accuracy score as a float.
        None on any error.
    Raises:
        This function should not raise any Exception.
    """
    n = 0
    for pred, true in zip(y_pred, y_true):
        if pred == true:
            n += 1
    if normalize == False:
        return n
    if y_pred.shape[0] > 0:
        return n / y_pred.shape[0]
    return None
