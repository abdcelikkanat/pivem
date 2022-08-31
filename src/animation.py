import os
import torch
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Animation:

    def __init__(self, embs, time_list, group_labels, dataset=None,
                 colors=None, color_name="Colors", sizes=None, title="",
                 padding=0.1, figure_path=None):

        self.__df = None
        self.__embs = embs.reshape(-1, 2)
        # self.__embs_node_idx = np.tile(np.arange(embs.shape[1]), (embs.shape[0], 1)).flatten()
        self.__time_list = time_list
        self.__frame_times = np.unique(self.__time_list)
        self.__num_of_frames = len(self.__frame_times)
        self.__num_of_nodes = int(self.__embs.shape[0] / self.__num_of_frames)

        self.__group_labels = group_labels
        self.__dataset = dataset
        self.__colors = colors if colors is not None else self.__group_labels
        self.__color_name = color_name
        self.__sizes = sizes if sizes is not None else [10] * self.__embs.shape[0]
        self.__title = title
        self.__padding = padding
        self.__figure_path = figure_path
        self.__run()

    def __get_dataframe(self):

        df = pd.DataFrame({"x-axis": self.__embs[:, 0],
                           "y-axis": self.__embs[:, 1],
                           "time": self.__time_list,
                           "group": self.__group_labels,
                           self.__color_name: self.__colors,
                           "size": self.__sizes,
                           })

        return df

    def __run(self):

        df = self.__get_dataframe()

        range_x = [df["x-axis"].min() - self.__padding, df["x-axis"].max() + self.__padding]
        range_y = [df["y-axis"].min() - self.__padding, df["y-axis"].max() + self.__padding]


        df_edges = dict()
        node0_idx = np.arange(self.__num_of_frames) * self.__num_of_nodes
        node1_idx = np.arange(self.__num_of_frames) * self.__num_of_nodes + 1

        plt.figure()
        fig = px.scatter(df, x="x-axis", y="y-axis", animation_frame="time", size=df["size"], animation_group="group",
                         color=self.__color_name,
                         color_discrete_sequence=px.colors.qualitative.Pastel,
                         size_max=10, title=self.__title, range_x=range_x, range_y=range_y)

        fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 100
        fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 1
        fig.update_layout(plot_bgcolor='white')
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='LightGray')
        fig.update_traces(marker=dict(size=20, line=dict(width=1, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))

        if self.__dataset is not None:
            data = self.__embs.reshape(self.__num_of_frames, self.__num_of_nodes, 2)
            df_edges = {'x': [], 'y': [], 'time': [], 'color': [], 'alpha': []}
            for f in range(self.__num_of_frames):
                for i in range(self.__num_of_nodes):
                    for j in range(i+1, self.__num_of_nodes):
                        df_edges['x'].extend([ data[f, i, 0], data[f, j, 0] ])
                        df_edges['y'].extend([ data[f, i, 1], data[f, j, 1] ])
                        df_edges['time'].extend([ f, f ])
                        df_edges['color'].extend([ i*self.__num_of_nodes+j, i*self.__num_of_nodes+j ])
                        df_edges['alpha'].extend([ 'rgba(0, 0, 0, 0.05)', 'rgba(0, 0, 0, 0.05)' ])
            df_edges = pd.DataFrame(df_edges)

            # df_edges["x"] = np.vstack((self.__embs[node0_idx, 0], self.__embs[node0_idx, 0])).T.flatten()
            # df_edges["y"] = np.vstack((self.__embs[node1_idx, 1], self.__embs[node1_idx, 1])).T.flatten()
            # df_edges["time"] = np.tile(self.__frame_times.reshape(-1, 1), (1, self.__num_of_nodes)).flatten()
            #
            # fig.add_traces(px.line(df, x=df_edges["x"], y=df_edges["y"], animation_frame="time").data)
            #
            # fig_lines = px.line(df, x="x-axis", y="y-axis", animation_frame="time")
            #
            fig_lines = px.line(df_edges, x="x", y="y", animation_frame="time",  color="color",
                                color_discrete_sequence=df_edges['alpha'])
            fig_lines.update_layout(showlegend=False)

            fig = go.Figure(data=fig.data+fig_lines.data, frames=fig.frames, layout=fig.layout)
            for i in range(len(fig.frames)):
                fig.frames[i].data = fig.frames[i].data + fig_lines.frames[i].data
            fig.update_layout(showlegend=False)

        # plt.figure()
        # df = pd.DataFrame({"x": [0, 1, 2, 3],
        #                    "y": [2, 3, 4, 5],
        #                    "time": [1, 1, 2, 2],
        #                    })
        # # plt.figure()
        # fig2 = px.line(df, x="x", y="y", animation_frame="time")
        # fig2.update_layout(yaxis={"range": [0, 5]}, xaxis={"range": [0, 5]})
        # fig2.show()


        # print(fig)
        # self.__events=2
        # if self.__events is not None:

        # z = torch.arange(len(self.__time_list)).reshape(-1, 1)
        # fig.add_trace(fig.add_shape(type="line",
        #               x0=z*0,
        #               y0=z*0,
        #               x1=z*100,
        #               y1=z*200,
        #               line=dict(
        #                   color="LightSeaGreen",
        #                   width=4,
        #                   dash="dashdot",
        #               )
        #               )
        #            )
        # print(fig)

        if self.__figure_path is None:
            fig.show()
        else:
            fig.write_html(self.__figure_path, auto_open=False)
            # fig.write_image(self.__figure_path)

        plt.figure()

        # df2 = pd.DataFrame({"x-axis": [-1, 0, 1, 0, 0, 1, 0, -1],
        #                    "y-axis": [0, 1, 0, -1, -1, 0, 1, 0],
        #                    "time": [0, 1, 2, 3, 0, 1, 2, 3],
        #                    "size": 10,
        #                    })
        # fig2 = px.scatter(df2, x="x-axis", y="y-axis", animation_frame="time", size=df2["size"],
        #                  color_discrete_sequence=px.colors.qualitative.Pastel,
        #                  size_max=10, title=self.__title, range_x=[-1.5, 1.5], range_y=[-1.5, +1.5])
        #
        # fig3 = px.line(df2, x="x-axis", y="y-axis", animation_frame="time", range_x=[-1.5, 1.5], range_y=[-1.5, +1.5])
        # # fig2.add_traces(px.line(df, x=[-1, 0, 1, 0, 0, 1, 0, -1], y=[0, 1, 0, -1, -1, 0, 1, 0],
        # #                         animation_frame=[0, 1, 2, 3, 0, 1, 2, 3]).data)
        #
        # fig4 = go.Figure(data=fig2.data + fig3.data,  frames=fig2.frames, layout=fig2.layout)
        # fig4.frames[0].data = fig2.frames[0].data + fig3.frames[0].data
        # fig4.frames[1].data = fig2.frames[1].data + fig3.frames[1].data
        # fig4.frames[2].data = fig2.frames[2].data + fig3.frames[2].data
        # fig4.frames[3].data = fig2.frames[3].data + fig3.frames[3].data
        #
        # print("-----")
        # print(fig4.frames)
        # fig4.show()

# import os
# import torch
# import plotly.express as px
# import pandas as pd
# import matplotlib.pyplot as plt
#
# class Animation:
#
#     def __init__(self, df, title, anim_frame, anim_group, anim_size, anim_color, anim_hover_name):
#
#         self.__df = df
#         self.__anim_frame = anim_frame
#         self.__anim_group = anim_group
#         self.__anim_size = anim_size
#         self.__anim_color = anim_color
#         self.__anim_hover_name = anim_hover_name
#         self.__title = title
#         self.__colors = ['r', 'b', 'k', 'm']
#
#         self.__run()
#
#     def __run(self):
#         #print(self.__df["node_id"])
#         node_colors = [self.__colors[node_id] for node_id in self.__df["node_id"]]
#         import numpy as np
#         pad = 0.1
#         range_x = [ self.__df["x"].min() - pad, self.__df["x"].max() + pad ]
#         range_y = [ self.__df["y"].min() - pad, self.__df["y"].max() + pad ]
#         plt.figure()
#         fig = px.scatter(self.__df, x="x", y="y", animation_frame=self.__anim_frame, animation_group=self.__anim_group,
#                    size=[10]*self.__df.shape[0], color=node_colors, hover_name=self.__anim_hover_name,
#                    size_max=10, range_x=range_x, range_y=range_y)
#         #log_x=True,
#         plt.title(self.__title)
#         fig.show()
#
#