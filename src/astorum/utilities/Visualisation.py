import numpy as np
import hyperspy.api as hs
import matplotlib.pyplot as plt

def plot_data_model_ROI(dataset, G, W, H, elementList):   
    WH = np.matmul(W, H)

    contributions = [hs.signals.Signal1D((G[:, [i]] @ (WH)[[i], :]).T.reshape(dataset.data.shape)) for i in range(G.shape[1])]
    contributions.append(hs.signals.Signal1D((np.matmul(G, WH)).T.reshape(dataset.data.shape)))

    titles = elementList + ["Background 1", "Background 2", "Full Model"]
    for i, c in enumerate(contributions):
        for a, b in zip(c.axes_manager._axes, dataset.axes_manager._axes):
            a.update_from(b)
        c.metadata.General.title = titles[i]

    fig, ax = plt.subplots()
    dataset.plot()

    roi = hs.roi.RectangularROI(left = dataset.axes_manager[1].index, top = dataset.axes_manager[0].index, right = dataset.axes_manager[1].size, bottom = dataset.axes_manager[0].size)
    
    imr = roi.interactive(dataset, color = 'green').sum(axis = 0).sum(axis = 0)
    contributions_roi = [roi.interactive(g, None).sum(axis = 0).sum(axis = 0) for g in contributions]

    spectra = [imr] + contributions_roi
    lines = []
    
    line, = ax.plot(dataset.axes_manager.signal_axes[0].axis, imr.data, label = imr.metadata.General.title, linestyle = "-")
    lines.append(line)
    
    for spectrum in contributions_roi:
        line, = ax.plot(dataset.axes_manager.signal_axes[0].axis, spectrum.data, label = spectrum.metadata.General.title, linestyle = "--")
        lines.append(line)
    
    ax.legend()
    ax.set_xlabel("Energy (keV)")
    ax.set_ylabel("Intensity")
    ax.set_title("EDXS Model Fit")

    def update_plot(*args, **kwargs):
        imr = roi.interactive(dataset, color = 'green').sum(axis = 0).sum(axis = 0)
        contributions_roi = [roi.interactive(g, None).sum(axis = 0).sum(axis = 0) for g in contributions]

        all_data = [imr] + contributions_roi
        for line, new_data in zip(lines, all_data):
            line.set_ydata(new_data.data)

        ax.relim()
        ax.autoscale_view()
        fig.canvas.draw_idle()

    roi.events.changed.connect(update_plot)
    update_plot()
    plt.show()

    return