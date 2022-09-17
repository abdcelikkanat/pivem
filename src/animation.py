import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import seaborn as sns


class Animation:

    def __init__(self, embs, frame_times: np.asarray = None, data: tuple = (None, None),
                 figsize=(12, 10), node_sizes=100, node2color: list = None,
                 color_palette="rocket_r", padding=0.1, fps=6,):

        self._figsize = figsize
        self._anim = None
        self._fps = fps

        # Data properties
        self._embs = embs
        self._frame_times = frame_times
        self._event_pairs = data[0]
        self._event_times = data[1]
        self._frames_num = embs.shape[0]
        self._nodes_num = embs.shape[1]
        self._dim = embs.shape[2]

        # Visual properties
        sns.set_theme(style="ticks")
        node2color = [0]*self._nodes_num if node2color is None else node2color
        self._color_num = 1 if node2color is None else len(set(node2color))
        self._palette = sns.color_palette(color_palette, self._color_num)
        self._node_colors = [self._palette.as_hex()[node2color[node]] for node in range(self._nodes_num)]
        self._node_sizes = [node_sizes]*self._nodes_num if type(node_sizes) is int else node_sizes
        self._linewidths = 1
        self._edgecolors = 'k'
        self._padding = padding
        self._decay_coeff = 25

    # def __fix(self):
    #
    #     chosen_events = []
    #     for pair_events in self._event_times:
    #         for k in range(len(self._time_list)):
    #             idx = np.digitize(x=pair_events, bins=self._time_list, right=True)
    #             chosen_pair_events = [pair_events[idx == k].max() if k in idx else None]
    #             chosen_events.append(chosen_pair_events)

    def _render(self, fig, repeat=False):
        global sc, ax

        def __set_canvas():

            xy_min = self._embs.min(axis=0, keepdims=False).min(axis=0, keepdims=False)
            xy_max = self._embs.max(axis=0, keepdims=False).max(axis=0, keepdims=False)
            xlen_padding = (xy_max[0] - xy_min[0]) * self._padding
            ylen_padding = (xy_max[1] - xy_min[1]) * self._padding
            ax.set_xlim([xy_min[0] - xlen_padding, xy_max[0] + xlen_padding])
            ax.set_ylim([xy_min[1] - ylen_padding, xy_max[1] + ylen_padding])

        def __init_func():
            global sc, ax

            sc = ax.scatter(
                [0]*self._nodes_num, [0]*self._nodes_num,
                s=self._node_sizes, c=self._node_colors,
                linewidths=self._linewidths, edgecolors=self._edgecolors
            )

            __set_canvas()

        def __func(f):
            global sc, ax

            for line in list(ax.lines):
                ax.lines.remove(line)

            # __set_canvas()

            # Plot the nodes
            sc.set_offsets(np.c_[self._embs[f, :, 0], self._embs[f, :, 1]])

            # Plot the event links
            if self._event_times is not None and self._event_pairs is not None:

                for pair_events, pair in zip(self._event_times, self._event_pairs):
                    i, j = pair
                    diff = self._frame_times[f] - pair_events
                    weight = np.where(diff >= 0, diff, np.inf).min()
                    if weight < np.inf:
                        ax.plot(
                            [self._embs[f, i, 0], self._embs[f, j, 0]],
                            [self._embs[f, i, 1], self._embs[f, j, 1]],
                            color='k',
                            alpha=np.exp(-self._decay_coeff * weight)
                        )

        anim = animation.FuncAnimation(
            fig=fig, init_func=__init_func, func=__func, frames=self._frames_num, interval=100, repeat=repeat
        )

        return anim

    def save(self, filepath, format="mp4"):
        global sc, ax

        fig, ax = plt.subplots(figsize=self._figsize, frameon=True)
        ax.set_axis_off()
        x_min, y_min = self._embs.min(axis=0).min(axis=0)
        x_max, y_max = self._embs.max(axis=0).max(axis=0)
        self._anim = self._render(fig)

        # fig.set_size_inches(y_max-y_min, x_max-x_min, )
        if format == "mp4":
            writer = animation.FFMpegWriter(fps=self._fps)

        elif format == "gif":
            writer = animation.PillowWriter(fps=self._fps)

        else:
            raise ValueError("Invalid format!")

        self._anim.save(filepath, writer)


# embs = np.random.randn(100, 10, 2)
# anim = Animation(embs)
# anim.save("./deneme.mp4")