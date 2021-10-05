import numpy as np


def swap(a, b):
    return b, a


def dist_between(coor1, coor2):
    """
    Calculate distance between two coordinates.

    Parameters
    ----------
    coor1 : list (len = 2)
        Coordinates.
    coor2 : list (len = 2)
        Coordinates.

    Returns
    -------
    distance : float
        Distance between two coordinates.

    Examples
    --------
    >>> a = np.array([1,4])
    >>> b = np.array([4,8])
    >>> dist_between(a,b)
    5
    """
    return ((coor1[0] - coor2[0]) ** 2 + (coor1[1] - coor2[1]) ** 2) ** (1 / 2)


def interior_division(coor1, coor2, m, n):
    """
    Calculate internally dividing point.

    Parameters
    ----------
    coor1 : list (len = 2)
        Coordinates.
    coor2 : list (len = 2)
        Coordinates.
    m : float
        Ratio.
    n : float
        Ratio.

    Returns
    -------
    point : numpy.ndarray
        Internally dividing point.

    Examples
    --------
    >>> a = np.array([1,4])
    >>> b = np.array([4,10])
    >>> m = 1
    >>> n = 2
    >>> interior_division(a,b,m,n)
    np.array([2,6])
    """
    return np.array(
        [
            (n * coor1[0] + m * coor2[0]) / (m + n),
            (n * coor1[1] + m * coor2[1]) / (m + n),
        ]
    )


def rotation_point(target, center, theta):
    """
    Calculate rotated point.

    Parameters
    ----------
    target : list (len = 2)
        Coordinates.
    center : list (len = 2)
        Coordinates.
    theta : float
        Degree angle.

    Returns
    -------
    coor : numpy.ndarray
        Rotated point.

    Examples
    --------
    >>> a = np.array([2,0])
    >>> b = np.array([0,1])
    >>> theta = 90
    >>> rotation_point(a,b,theta)
    np.array([3,2])
    """
    b = np.array(target) - np.array(center)
    theta = theta * np.pi / 180
    s = np.sin(theta)
    c = np.cos(theta)
    rm = [[c, -s], [s, c]]
    coor = np.dot(rm, b) + np.array(center)
    return coor


def coors_to_linearfunction(coor1, coor2, form="general"):
    """
    Calculate coefficients of linearfunction from two coordinates.

    Parameters
    ----------
    target : list (len = 2)
        Coordinates.
    center : list (len = 2)
        Coordinates.
    form : str {general, slope-intercept}
        If you choose general, function is defined as ax+by+c=0.
        If you choose slope-intercept, function is defined as y=ax+b.

    Returns
    -------
    coef : list
        Coefficients of linearfunction.
        If you choose general, return [a,b,c].
        If you choose slope-intercept, return [a,b].

    Examples
    --------
    >>> a = np.array([2,4])
    >>> b = np.array([3,7])
    >>> coors_to_linearfunction(a,b,"slope-intercept")
    np.array([3,-2])
    """
    forms = ["general", "slope-intercept"]
    assert form in forms, f"{form} is not in {forms}."
    m = (coor2[1] - coor1[1]) / (coor2[0] - coor1[0])
    c = coor1[1] - m * coor1[0]
    if form == forms[0]:
        return [m, -1, c]
    elif form == forms[1]:
        return [m, c]


def dist_point_line(coor, coef):
    d = (abs(coef[0] * coor[0] + coef[1] * coor[1] + coef[2])) / (
        (coef[0] ** 2 + coef[1] ** 2) ** (1 / 2)
    )
    return d


def pixel_between(coor1, coor2):
    """
    Find coordinates on lines between two coordinates.

    Parameters
    ----------
    target : list (len = 2)
        Coordinates.
    center : list (len = 2)
        Coordinates.

    Returns
    -------
    coor : list
        Coordinates list.

    Examples
    --------
    >>> a = np.array([0,0])
    >>> b = np.array([9,5])
    >>> pixel_between(a,b)
    np.array([[0, 0], [1, 0], [1, 1], [2, 1], [3, 1], [3, 2], [4, 2],
    [5, 2], [5, 3], [6, 3], [7, 3], [7, 4], [8, 4], [8, 5], [9, 5]])
    """
    pixels = []
    if coor1[0] > coor2[0]:
        coor1, coor2 = swap(coor1, coor2)
    coef = coors_to_linearfunction(coor1, coor2, "slope-intercept")

    # print(coor)
    for x in range(int(coor1[0]), int(coor2[0]) + 1):
        y_forward = coef[0] * x + coef[1]
        y_backward = coef[0] * (x + 1) + coef[1]
        # print(x, y_forward, y_backward)
        if y_forward < y_backward:
            y_forward, y_backward = swap(y_forward, y_backward)
        y_forward = int(y_forward)
        y_backward = int(y_backward)
        # print(x, y_forward, y_backward)
        for y in range(y_backward, y_forward + 1):
            pixels.append([x, y])
    return pixels
