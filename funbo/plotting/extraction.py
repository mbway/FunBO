
def plot_acquisition(optimiser, surrogate, x, prev_xy=None, beta=1, l=1):
    fig = plt.figure(figsize=(16, 8))
    grid = gridspec.GridSpec(nrows=2, ncols=1, height_ratios=[2, 1])
    ax1, ax2 = fig.add_subplot(grid[0]), fig.add_subplot(grid[1])

    ymin, ymax = optimiser.range_bounds
    for ax in (ax1, ax2):
        margin = (ymax-ymin)*0.005
        ax.set_xlim((ymin-margin, ymax+margin))

    intervals = 100

    assert len(x.shape) == 1, 'x not 0-dimensional'
    xs = np.repeat(x.reshape(1, -1), intervals, axis=0) # stack copies of x as rows
    ys = np.linspace(*optimiser.range_bounds, num=intervals)

    mu, var = surrogate.predict(np.hstack((xs, ys.reshape(-1, 1))))

    # Surrogate plot
    ax1.plot(ys, mu, label='surrogate $\mu$')
    sig = np.sqrt(var)
    n_sig = 2
    ax1.fill_between(ys, (mu-n_sig*sig).flatten(), (mu+n_sig*sig).flatten(),
                     alpha=0.2, label=r'surrogate ${}\sigma$'.format(n_sig))
    ax1.set_xlabel('$f(x)$')
    ax1.set_ylabel('$R_l(x,f(x))$')
    ax1.legend()

    # Acquisition plot
    def get_X(ys):
        xs = np.repeat(x.reshape(1, -1), ys.shape[0], axis=0) # stack copies of x as rows
        return np.hstack((xs, ys))

    def acq(ys):
        return fbo.UCB(get_X(ys), beta=beta, surrogate=surrogate, maximising=optimiser.is_maximising())
    UCBs = acq(ys.reshape(-1, 1))
    UCB_min = np.min(UCBs)
    UCBs -= UCB_min # shift to sit on x axis

    ax2.plot(ys, UCBs, color='C1', label=r'$\alpha$')
    best_y, info = fbo.maximise(acq, (ymin, ymax))
    ax2.plot(best_y, info['max_acq']-UCB_min, 'o', color='orange', label=r'$\max\,\alpha$')

    if prev_xy is not None:
        ws = fbo.tracking_weights(get_X(ys.reshape(-1, 1)), prev_xy, l=l)
        ax2.plot(ys, ws*0.4*np.max(UCBs), color='C2', label=r'$k((x_{prev},f(x_{prev})), (x,f(x)))$ (rescaled)')

        ax2.plot(ys, UCBs*ws, color='C3', label=r'$k\alpha$')
        def tracked_acq(ys):
            X = get_X(ys)
            UCBs = fbo.UCB(X, beta=beta, surrogate=surrogate, maximising=optimiser.is_maximising())
            UCBs -= np.min(UCBs) # shift to sit on x axis
            ws = fbo.tracking_weights(X, prev_xy, l=l)
            return UCBs*ws

        best_y, info = fbo.maximise(tracked_acq, (ymin, ymax))
        ax2.plot(best_y, info['max_acq'], 'ro', label=r'$\max\,k\alpha$')

    for ax in (ax1, ax2):
        ax.axvline(best_y, linestyle='--', color=(1.0, 0.3, 0.3), alpha=0.4, zorder=-1)
        prev_y = prev_xy[0,-1]
        ax.axvline(x=prev_y, linestyle=':', color='grey', alpha=0.4, zorder=-1, label='$f(x_{prev})$' if ax == ax2 else None)

    ax2.set_xlabel('$f(x)$')
    ax2.set_ylabel(r'$\alpha(f(x))$')
    ax2.legend()

    fig.tight_layout()

