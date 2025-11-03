import numpy as np

def runningmedian(x, y, nbins=15, stat='median'):
    """
    Compute running median (or mean) and scatter for y as a function of x.
    Returns: bin_centers, y_median, y_low, y_high, n_points
    """

    if len(x) == 0:
        return np.array([]), np.array([]), np.array([]), np.array([]), np.array([])
    
    x = np.array(x)
    y = np.array(y)
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]

    bins = np.linspace(np.min(x), np.max(x), nbins + 1)
    bin_cent = 0.5 * (bins[:-1] + bins[1:])
    ymean, ysiglo, ysigup, ndata = [], [], [], []

    for i in range(len(bins) - 1):
        mask = (x >= bins[i]) & (x < bins[i + 1])
        vals = y[mask]
        if len(vals) > 0:
            if stat == 'median':
                med = np.median(vals)
                low = np.percentile(vals, 16)
                high = np.percentile(vals, 84)
            else:
                med = np.mean(vals)
                low = med - np.std(vals)
                high = med + np.std(vals)
            ymean.append(med)
            ysiglo.append(low)
            ysigup.append(high)
            ndata.append(len(vals))
        else:
            ymean.append(np.nan)
            ysiglo.append(np.nan)
            ysigup.append(np.nan)
            ndata.append(0)

    return np.array(bin_cent), np.array(ymean), np.array(ysiglo), np.array(ysigup), np.array(ndata)