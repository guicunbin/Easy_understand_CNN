def np_array_equal(a1, a2):
    assert (len(a1.shape) == 1 and len(a2.shape) == 1);
    for i in range(len(a1)):
        if((a1[i] > a2[i] + 1e-4) or (a1[i] < a2[i] - 1e-4)):
            return False;
    return True;
