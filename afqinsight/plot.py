"""Create diagnostic plots of AFQ-Insight output."""
# import itertools
# import os.path as op

# import matplotlib.pyplot as plt
# import numpy as np
# import palettable
# from bokeh.embed import file_html
# from bokeh.layouts import column, row
# from bokeh.models import BoxAnnotation, ColorBar, CustomJS, HoverTool
# from bokeh.models import Label, Range1d, Title
# from bokeh.models.mappers import LinearColorMapper
# from bokeh.models.tickers import FixedTicker
# from bokeh.palettes import Category10, Spectral10, Cividis256
# from bokeh.plotting import ColumnDataSource, figure, show
# from bokeh.resources import CDN
# from matplotlib.colors import to_hex
# from sklearn.decomposition import PCA

# from . import utils


# def plot_betas(
#     beta_hat,
#     columns,
#     ecdf=False,
#     output_html=None,
#     width=750,
#     height=250,
#     sizing_mode="stretch_both",
# ):
#     """Plot the classification probabilities for each cross-validation split

#     Parameters
#     ----------
#     beta_hat : OrderedDict of beta coeffs
#         Two-level ordered dict with beta_hat coefficients, ordered first
#         by tract and then by metric

#     columns : pd.MultiIndex
#         MultiIndex columns of the feature matrix

#     ecdf : bool
#         If True, plot the estimated cumulative probability distribution (ECDF)
#         of the beta coefficients. If False, plot the raw coefficients.

#     output_html : string or None, default=None
#         Filename for bokeh html output. If None, figure will not be saved

#     width : int, default=750
#         Width of each beta plot (in pixels)

#     height : int, default=250
#         Height of each beta plot (in pixels)

#     sizing_mode : string
#         One of ("fixed", "stretch_both", "scale_width", "scale_height",
#         "scale_both"). Specifies how will the items in the layout resize to
#         fill the available space. Default is "stretch_both". For more
#         information on the different modes see
#         https://bokeh.pydata.org/en/latest/docs/reference/models/layouts.html#bokeh.models.layouts.LayoutDOM

#     See Also
#     --------
#     transform.AFQFeatureTransformer
#         Transforms AFQ csv files into feature matrix. Use this to create
#         the `columns` input.

#     transform.beta_hat_by_groups
#         Returns a two-level ordered dict instead of "unfolding" the tracts
#     """
#     all_metrics = columns.levels[columns.names.index("metric")]
#     ps = []

#     colors = {}
#     for idx, metric in enumerate(all_metrics):
#         colors[metric] = Spectral10[idx]

#     for idx, tract in enumerate(beta_hat.keys()):
#         ps.append(
#             figure(plot_width=width, plot_height=height, toolbar_location="right")
#         )
#         ps[idx].title.text = tract

#         for metric in beta_hat[tract].keys():
#             b = beta_hat[tract][metric]
#             if not all(b == 0):
#                 if ecdf:
#                     cdf = utils.ecdf(b)
#                     ps[idx].circle(
#                         cdf.x,
#                         cdf.y,
#                         size=5,
#                         color=colors[metric],
#                         alpha=0.8,
#                         legend=metric,
#                     )
#                 else:
#                     ps[idx].line(
#                         np.arange(len(b)),
#                         b,
#                         line_width=4,
#                         color=colors[metric],
#                         alpha=0.8,
#                         legend=metric,
#                     )

#         ps[idx].legend.location = "bottom_right"
#         ps[idx].legend.click_policy = "hide"

#     p = column(ps)
#     p.sizing_mode = sizing_mode

#     if output_html is not None:
#         html = file_html(p, CDN, "my plot")
#         with open(op.abspath(output_html), "w") as fp:
#             fp.write(html)
#     else:
#         show(p)


# def plot_classification_probabilities(
#     x,
#     y,
#     cv_results,
#     output_html=None,
#     width=500,
#     height=500,
#     sizing_mode="stretch_both",
# ):
#     """Plot the classification probabilities for each cross-validation split

#     Parameters
#     ----------
#     x : numpy.ndarray
#         The original feature matrix

#     y : numpy.ndarray
#         The target array (i.e. "ground truth")

#     cv_results : list of SGLResult namedtuples
#         Results of each cross-validation split

#     output_html : string or None, default=None
#         Filename for bokeh html output. If None, figure will not be saved

#     width : int, default=500
#         Width of each beta plot (in pixels)

#     height : int, default=500
#         Height of each beta plot (in pixels)

#     sizing_mode : string
#         One of ("fixed", "stretch_both", "scale_width", "scale_height",
#         "scale_both"). Specifies how will the items in the layout resize to
#         fill the available space. Default is "stretch_both". For more
#         information on the different modes see
#         https://bokeh.pydata.org/en/latest/docs/reference/models/layouts.html#bokeh.models.layouts.LayoutDOM
#     """
#     p = figure(plot_width=width, plot_height=height, toolbar_location="above")
#     p.title.text = "Classification probabilities for each CV split"
#     p.add_layout(
#         Title(
#             text="Click on legend entries to hide/show corresponding lines",
#             align="left",
#         ),
#         "right",
#     )

#     names = ["cv_idx = {i:d}".format(i=i) for i in range(len(cv_results))]

#     hover = HoverTool(tooltips=[("Subject", "$index")], mode="vline")
#     hover.point_policy = "snap_to_data"
#     hover.line_policy = "nearest"

#     for res, color, name in zip(cv_results, Spectral10, names):
#         p.line(
#             np.arange(len(y)),
#             _sigmoid(x.dot(res.beta_hat)),
#             line_width=2,
#             color=color,
#             alpha=0.8,
#             legend=name,
#         )

#     p.line(np.arange(len(y)), y, line_width=3, alpha=0.8, legend="ground truth")
#     p.line(
#         np.arange(len(y)),
#         0.5 * np.ones(len(y)),
#         line_width=2,
#         line_dash="dashed",
#         alpha=0.8,
#         legend="threshold",
#     )
#     p.add_tools(hover)
#     p.legend.location = "top_right"
#     p.legend.click_policy = "hide"
#     p.xaxis.axis_label = "Subject ID"
#     p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
#     p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
#     p.xaxis.major_label_text_font_size = "0pt"  # turn off x-axis tick labels
#     p.yaxis.axis_label = "Classification Probability"
#     p.sizing_mode = sizing_mode

#     if output_html is not None:
#         html = file_html(p, CDN, "my plot")
#         with open(op.abspath(output_html), "w") as fp:
#             fp.write(html)
#     else:
#         show(p)


# def plot_unfolded_beta(
#     unfolded_beta,
#     output_html=None,
#     width=500,
#     height=500,
#     sizing_mode="stretch_both",
#     tract_names=None,
# ):
#     """Plot the regression coefficients on the "unfolded" brain.

#     All tracts are represented contiguously on the x-axis in this order:
#     - left hemisphere (lateral to medial)
#     - callosal tracts (anterior then posterior)
#     - Right hemisphere (medial to lateral)
#     Each metric is plotted separately

#     Parameters
#     ----------
#     unfolded_beta : dict
#         Dictionary of beta arrays, where the keys are the diffusion metrics
#         (e.g. 'fa', 'md') and the values are the contiguous regression
#         coefficient arrays

#     output_html : string or None, default=None
#         Filename for bokeh html output. If None, figure will not be saved

#     width : int, default=500
#         Width of each beta plot (in pixels)

#     height : int, default=500
#         Height of each beta plot (in pixels)

#     sizing_mode : string
#         One of ("fixed", "stretch_both", "scale_width", "scale_height",
#         "scale_both"). Specifies how will the items in the layout resize to
#         fill the available space. Default is "stretch_both". For more
#         information on the different modes see
#         https://bokeh.pydata.org/en/latest/docs/reference/models/layouts.html#bokeh.models.layouts.LayoutDOM

#     tract_names : list or None, default=None
#         Names of the tracts. If None, use utils.CANONICAL_TRACT_NAMES
#     """
#     p = figure(plot_width=width, plot_height=height, toolbar_location="above")
#     p.title.text = 'Feature weights in the "unfolded" brain'
#     p.add_layout(
#         Title(
#             text="Click on legend entries to hide/show corresponding lines",
#             align="left",
#         ),
#         "right",
#     )

#     if tract_names is not None:
#         tracts = tract_names
#     else:
#         tracts = utils.CANONICAL_TRACT_NAMES

#     len_alltracts = len(unfolded_beta[list(unfolded_beta.keys())[0]])
#     n_tracts = len(tracts)
#     n_nodes = len_alltracts / n_tracts
#     ticks = np.arange(0, len_alltracts, n_nodes)

#     p.xaxis.major_tick_line_color = None  # turn off x-axis major ticks
#     p.xaxis.minor_tick_line_color = None  # turn off x-axis minor ticks
#     p.xaxis.major_label_text_font_size = "0pt"  # turn off x-axis tick labels

#     p.x_range = Range1d(0, len_alltracts)
#     p.xgrid.grid_line_color = None

#     tract_colors = np.copy(palettable.tableau.Tableau_20.hex_colors).tolist()
#     tract_colors = tract_colors[1:-2:2] + tract_colors[-2:] + tract_colors[-3::-2]

#     for i, color in enumerate(tract_colors):
#         p.add_layout(
#             BoxAnnotation(
#                 left=ticks[i],
#                 right=ticks[i] + n_nodes,
#                 fill_color=color,
#                 fill_alpha=0.3,
#             )
#         )

#         p.add_layout(
#             Label(
#                 x=ticks[i] + n_nodes / 2,
#                 y=10,
#                 x_units="data",
#                 y_units="screen",
#                 text=tracts[i],
#                 render_mode="canvas",
#                 border_line_alpha=0.0,
#                 background_fill_alpha=0.0,
#                 angle=np.pi / 2,
#                 text_align="left",
#                 text_baseline="middle",
#             )
#         )

#     unfolded_beta["x"] = np.arange(len_alltracts)
#     source = ColumnDataSource(data=unfolded_beta)
#     del unfolded_beta["x"]

#     lines = {}
#     for name, color in zip(
#         unfolded_beta.keys(),
#         Category10[max(len(list(unfolded_beta.keys())), min(list(Category10.keys())))],
#     ):
#         lines[name] = p.line(
#             x="x",
#             y=name,
#             source=source,
#             name=name,
#             line_width=3,
#             color=color,
#             legend=dict(value=name),
#         )

#     p.add_tools(
#         HoverTool(
#             show_arrow=False,
#             line_policy="interp",
#             tooltips=[("metric", "$name"), ("beta", "$y")],
#             point_policy="snap_to_data",
#             renderers=list(lines.values()),
#         )
#     )

#     p.legend.location = "center_right"
#     p.legend.click_policy = "hide"

#     p.sizing_mode = sizing_mode

#     if output_html is not None:
#         html = file_html(p, CDN, "my plot")
#         with open(op.abspath(output_html), "w") as fp:
#             fp.write(html)
#     else:
#         show(p)


# def plot_pca_space(
#     x,
#     y,
#     beta,
#     plot_type,
#     target_name,
#     classification_strings=None,
#     plot_both=True,
#     output_html=None,
#     width=500,
#     height=500,
#     sizing_mode="stretch_both",
# ):
#     """Plot regression predictions in a 2-component PCA space.

#     This function has two plot modes, specified by the `plot_both` flag. If
#     `plot_both` == True, this plots side-by-side scatter plots of the target
#     variable in 2-D PCA space. The right plot is the post-SGL weighted feature
#     matrix and the left plot is the pre-SGL original feature matrix.

#     Otherwise this plots only the post-SGL weighted feature space and also
#     plots a contour of the regression prediction.

#     Parameters
#     ----------
#     x : numpy.ndarray
#         Feature matrix

#     y : pandas.Series
#         Binary classification target array

#     beta : numpy.ndarray
#         Regression coefficients

#     plot_type : 'regression' or 'classification'
#         Type of ML problem

#     target_name : string
#         The name of the target variable (used in hover tool)

#     classification_strings : dict
#         Dictionary mapping the categorical numerical target values onto their
#         names. If `plot_type` == "regression", this parameter is not used.

#     plot_both : boolean, default=True
#         If True, plot the PCA in both the original feature space and the
#         feature space projected onto the coefficient vector

#     output_html : string or None, default=None
#         Filename for bokeh html output. If None, figure will not be saved

#     width : int, default=500
#         Width of each beta plot (in pixels)

#     height : int, default=500
#         Height of each beta plot (in pixels)

#     sizing_mode : string
#         One of ("fixed", "stretch_both", "scale_width", "scale_height",
#         "scale_both"). Specifies how will the items in the layout resize to
#         fill the available space. Default is "stretch_both". For more
#         information on the different modes see
#         https://bokeh.pydata.org/en/latest/docs/reference/models/layouts.html#bokeh.models.layouts.LayoutDOM
#     """
#     if plot_type not in ["regression", "classification"]:
#         raise ValueError(
#             '`plot_type` must be either "classification" or ' '"regression"'
#         )

#     x_projection = np.outer(x.dot(beta), beta) / (np.linalg.norm(beta) ** 2.0)

#     pca_orig = PCA(n_components=2)
#     pca_sgl = PCA(n_components=2)

#     x2_sgl = pca_sgl.fit_transform(x_projection)
#     x2_orig = pca_orig.fit_transform(x)

#     if plot_type == "classification":
#         cmap = plt.get_cmap("RdBu")
#         colors = [to_hex(c) for c in cmap(np.linspace(1, 0, 256))]
#     else:
#         colors = Cividis256

#     color_mapper = LinearColorMapper(palette=colors, low=np.min(y), high=np.max(y))

#     color_bar = ColorBar(
#         color_mapper=color_mapper,
#         ticker=FixedTicker(ticks=np.arange(np.min(y), np.max(y), 5)),
#         label_standoff=8,
#         border_line_color=None,
#         location=(0, 0),
#     )

#     if plot_type == "classification":
#         target = y.copy()
#         for k, v in classification_strings.items():
#             target[y == k] = v
#     else:
#         target = y.copy()

#     pc_info = {
#         "pc0_sgl": x2_sgl[:, 0],
#         "pc1_sgl": x2_sgl[:, 1],
#         "target": y.values,
#         "target_string": target.values,
#         "subject_id": target.index,
#     }

#     ps = [None]

#     if plot_both:
#         pc_info["pc0_orig"] = x2_orig[:, 0]
#         pc_info["pc1_orig"] = x2_orig[:, 1]
#         ps = [None] * 2

#     tooltips = [("Subject", "@subject_id"), (target_name, "@target_string")]

#     source = ColumnDataSource(data=pc_info)
#     code = "source.set('selected', cb_data.index);"
#     callback = CustomJS(args={"source": source}, code=code)

#     if not plot_both:
#         ps[0] = figure(
#             plot_width=int(width * 1.1), plot_height=height, toolbar_location="right"
#         )

#         npoints = 200
#         dx = np.max(x2_sgl[:, 0]) - np.min(x2_sgl[:, 0])
#         xmid = 0.5 * (np.max(x2_sgl[:, 0]) + np.min(x2_sgl[:, 0]))
#         xmin = xmid - (dx * 1.1 / 2.0)
#         xmax = xmid + (dx * 1.1 / 2.0)

#         dy = np.max(x2_sgl[:, 1]) - np.min(x2_sgl[:, 1])
#         ymid = 0.5 * (np.max(x2_sgl[:, 1]) + np.min(x2_sgl[:, 1]))
#         ymin = ymid - (dy * 1.1 / 2.0)
#         ymax = ymid + (dy * 1.1 / 2.0)

#         x_subspace = np.linspace(xmin, xmax, npoints)
#         y_subspace = np.linspace(ymin, ymax, npoints)
#         subspace_pairs = np.array(
#             [[p[0], p[1]] for p in itertools.product(x_subspace, y_subspace)]
#         )
#         bigspace_pairs = (
#             pca_sgl.inverse_transform(subspace_pairs) * np.linalg.norm(beta) ** 2.0
#         )
#         predict_pairs = bigspace_pairs.dot(
#             np.divide(
#                 np.ones_like(beta), beta, out=np.zeros_like(beta), where=beta != 0
#             )
#         )
#         x_grid, _ = np.meshgrid(x_subspace, y_subspace)
#         p_grid = predict_pairs.reshape(x_grid.shape, order="F")

#         ps[0].image(
#             image=[p_grid], x=xmin, y=ymin, dw=dx * 1.1, dh=dy * 1.1, palette=colors
#         )

#         ps[0].add_layout(color_bar, "right")
#         ps[0].x_range = Range1d(xmin, xmax)
#         ps[0].y_range = Range1d(ymin, ymax)
#     else:
#         ps[0] = figure(plot_width=width, plot_height=height, toolbar_location="right")

#         if plot_type == "regression":
#             ps[0].add_layout(color_bar, "right")

#     if plot_type == "regression":
#         ps[0].title.text = "Regression in Post-SGL PCA space"
#         s0 = ps[0].scatter(
#             "pc0_sgl",
#             "pc1_sgl",
#             source=source,
#             size=20,
#             fill_color={"field": "target", "transform": color_mapper},
#             line_color="white",
#             line_width=2.5,
#         )
#     else:
#         ps[0].title.text = "Classification in Post-SGL PCA space"
#         s0 = ps[0].scatter(
#             "pc0_sgl",
#             "pc1_sgl",
#             source=source,
#             size=20,
#             fill_color={"field": "target", "transform": color_mapper},
#             line_color="white",
#             line_width=2.5,
#             legend="target_string",
#         )

#     hover0 = HoverTool(tooltips=tooltips, callback=callback, renderers=[s0])
#     ps[0].add_tools(hover0)

#     if plot_both:
#         ps[1] = figure(plot_width=width, plot_height=height, toolbar_location="right")

#         if plot_type == "regression":
#             ps[1].title.text = "Regression in Original PCA space"
#             s1 = ps[1].scatter(
#                 "pc0_orig",
#                 "pc1_orig",
#                 source=source,
#                 size=20,
#                 fill_color={"field": "target", "transform": color_mapper},
#                 line_color="white",
#                 line_width=2.5,
#             )
#         else:
#             ps[1].title.text = "Classification in Original PCA space"
#             s1 = ps[1].scatter(
#                 "pc0_orig",
#                 "pc1_orig",
#                 source=source,
#                 size=20,
#                 fill_color={"field": "target", "transform": color_mapper},
#                 line_color="white",
#                 line_width=2.5,
#                 legend="target_string",
#             )

#         hover1 = HoverTool(tooltips=tooltips, callback=callback, renderers=[s1])
#         ps[1].add_tools(hover1)

#     for plot in ps:
#         plot.xaxis.axis_label = "1st Principal Component"
#         plot.yaxis.axis_label = "2nd Principal Component"

#     if plot_both:
#         layout = row(ps[::-1])
#     else:
#         layout = ps[0]

#     layout.sizing_mode = sizing_mode

#     if output_html is not None:
#         html = file_html(layout, CDN, "my plot")
#         with open(op.abspath(output_html), "w") as fp:
#             fp.write(html)
#     else:
#         show(layout)
