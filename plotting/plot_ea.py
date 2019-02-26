# <editor-fold desc="some imports">
from datetime import datetime
from ea.GenerationManager import PlaybackGenManager
import colorsys
import math
import numpy as np
# </editor-fold>


# <editor-fold desc="matplotlib imports">
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.font_manager import FontProperties as mFP


# </editor-fold>


# <editor-fold desc="generation history plotting">
def generation_history(genManager, filename, gen_per_plot=10, s=1, lv=.7, hsv_hls=False, NUM_COLORS=100):
  legend_scale = 20

  class HandlerCrossover(object):
    @staticmethod
    def get_patch(x=0, y=0, scale=1):
      return mpatches.RegularPolygon([x, y], 3, scale * 0.25, edgecolor='black', facecolor=(.54, .27, 0),
                                     label='Crossover')

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
      x0, y0 = handlebox.xdescent, handlebox.ydescent
      width, height = handlebox.width, handlebox.height
      center = 0.5 * width - 0.5 * x0, 0.2 * height - 0.5 * y0
      patch = HandlerCrossover.get_patch(center[0], center[1], legend_scale)
      patch.set_transform(handlebox.get_transform())
      handlebox.add_artist(patch)
      return patch

  class LegendCrossover(object):
    pass

  class HandlerMuation(object):
    @staticmethod
    def get_patch(x=0, y=0, scale=1):
      return mpatches.RegularPolygon([x, y], 6, scale * 0.25, edgecolor='black', facecolor=(.15, .15, .43),
                                     label='Mutation')

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
      x0, y0 = handlebox.xdescent, handlebox.ydescent
      width, height = handlebox.width, handlebox.height
      center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
      patch = HandlerMuation.get_patch(center[0], center[1], legend_scale)
      patch.set_transform(handlebox.get_transform())
      handlebox.add_artist(patch)
      return patch

  class LegendMutation(object):
    pass

  class HandlerIndividual(object):
    @staticmethod
    def get_patch(x=0, y=0, scale=1):
      return mpatches.Circle([x, y], scale * 0.25, edgecolor='black', facecolor=(0, 0, 0, 0),
                             label='Individual')

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
      x0, y0 = handlebox.xdescent, handlebox.ydescent
      width, height = handlebox.width, handlebox.height
      center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
      patch = HandlerIndividual.get_patch(center[0], center[1], legend_scale)
      patch.set_transform(handlebox.get_transform())
      handlebox.add_artist(patch)
      return patch

  class LegendIndividual(object):
    pass

  class HandlerUnusedIndividual(object):
    color = (.7, .7, .7)

    @staticmethod
    def get_patch(x=0, y=0, scale=1):
      p = HandlerIndividual.get_patch(x, y, scale)
      p.set_color(HandlerUnusedIndividual.color)
      p.set_label('Unused Individual')
      return p

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
      x0, y0 = handlebox.xdescent, handlebox.ydescent
      width, height = handlebox.width, handlebox.height
      center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
      patch = HandlerUnusedIndividual.get_patch(center[0], center[1], legend_scale)
      patch.set_transform(handlebox.get_transform())
      handlebox.add_artist(patch)
      return patch

  class LegendUnusedIndividual(object):
    pass

  class HandlerSelectedIndividual(object):
    @staticmethod
    def get_patch(x=0, y=0, scale=1.414):
      return mpatches.RegularPolygon([x, y], 4, scale * 0.25, edgecolor='black', facecolor=(0, 0, 0, 0),
                                     label='Selected Individual')

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
      x0, y0 = handlebox.xdescent, handlebox.ydescent
      width, height = handlebox.width, handlebox.height
      center = 0.5 * width - 0.5 * x0, 0.5 * height - 0.5 * y0
      patch = HandlerSelectedIndividual.get_patch(center[0], center[1], legend_scale)
      patch.set_transform(handlebox.get_transform())
      handlebox.add_artist(patch)
      return patch

  class LegendSelectedIndividual(object):
    pass

  def patch_path(x0=0, y0=0, x1=0, y1=0, width=.02):
    result = mpatches.FancyArrow(x0, y0, x1 - x0, y1 - y0, width=width, head_length=0, head_width=0)
    result.set_color('black')
    result.zorder = -1
    return result

  # colors = [colorsys.hls_to_rgb(i / NUM_COLORS, lv, s) if hsv_hls else colorsys.hsv_to_rgb(i / NUM_COLORS, s, lv) for i
  #           in range(NUM_COLORS)]

  p_ = np.asarray(plt.get_cmap('nipy_spectral')(np.linspace(.0, .95, NUM_COLORS)))
  p_ = [colorsys.rgb_to_hsv(i[0], i[1], i[2]) for i in p_]
  p_ = np.asarray([colorsys.hsv_to_rgb(i[0], .8 * i[1], .8 * i[2]) for i in p_])
  p_ = p_ * 255
  colors = ['#%02x%02x%02x' % (int(i[0]), int(i[1]), int(i[2])) for i in p_]

  unused_path_color = (.5, .5, .5)

  legend_elements = [LegendIndividual(), LegendUnusedIndividual(),
                     LegendSelectedIndividual(), LegendCrossover(),
                     LegendMutation()]

  y_step = -1
  x_step = -1
  x_init_offset = 2 * x_step
  prev_colors = {}

  plot_idx = 0
  fig = plt.figure()
  ax = fig.gca()
  fontP = mFP()
  fontP.set_size('small')
  plt.legend(legend_elements,
             ['Individual',
              'Unused Individual',
              'Selected Individual',
              'Crossover',
              'Mutation'],
             handler_map={LegendIndividual: HandlerIndividual(),
                          LegendUnusedIndividual: HandlerUnusedIndividual(),
                          LegendSelectedIndividual: HandlerSelectedIndividual(),
                          LegendCrossover: HandlerCrossover(),
                          LegendMutation: HandlerMuation()},
             ncol=5,
             loc='upper center',
             # bbox_to_anchor=(0.5,1.03),
             prop=fontP,
             framealpha=1, )

  bars_x = []
  bars_y = []
  bars_c = []
  bars_h = []
  color_dict = {}

  y_offset = 0

  generation_idx = []
  generation_label = []
  max_x_offset = 0

  # iterate over genrations
  for genMan in genManager:
    gen_acc = {}
    sel_acc = {}
    curr_sel_colors = {}
    x_coord_dict = {}
    x_offset = x_init_offset
    y_init_offset = y_offset
    genMan_descendants = genMan.descendants
    genMan_selection = genMan.selection
    genMan_generation = genMan.generation
    mutation_dict = dict(genMan.mutation_ancestor)

    for ind in [i for i in genMan_generation if i not in genMan_descendants]:
      color = color_dict.get(ind)
      if color != None:
        prev_colors[ind] = color

    free_colors = [c for c in colors if c not in prev_colors.values()]
    color_dict = prev_colors.copy()

    height = 2 if len(genMan_selection) > 0 and len(mutation_dict) > 0 else 0
    for parents, descendants in genMan.crossover_ancestor:
      width = 0
      descendants_coords = []
      dead_descendants_coords = []
      tmp_coord_dict = {}
      for descendant in descendants:
        if descendant in mutation_dict and not descendant in genMan_generation:
          mutation_breed = mutation_dict.get(descendant, [])
          _w = len(mutation_breed)
          mut_x, mut_y = x_offset + width * x_step + (_w - 1) / 2 * x_step, y_offset + 2 * y_step
          descendants_coords.append((mut_x, mut_y))
          patch = HandlerMuation.get_patch(mut_x, mut_y)
          ax.add_patch(patch)

          x_ = x_offset + width * x_step
          for m in mutation_breed:
            if m in genMan_generation:
              x_coord_dict[m] = (x_, (mut_x, mut_y))
            else:
              y_ = y_offset + 2.75 * y_step
              patch = HandlerUnusedIndividual.get_patch(x_, y_)
              ax.add_patch(patch)
              patch = patch_path(mut_x, mut_y, x_, y_)
              patch.set_color(unused_path_color)
              ax.add_patch(patch)
            x_ += x_step

          height = 3
          width += _w
          del mutation_dict[descendant]
          continue
        if descendant in genMan_generation:
          tmp_coord_dict[descendant] = (x_offset + width * x_step)
        else:
          x, y = x_offset + width * x_step, y_offset + 2.0 * y_step
          dead_descendants_coords.append((x, y))
          patch = HandlerUnusedIndividual.get_patch(x, y)
          ax.add_patch(patch)
        width += 1

      cross_x, cross_y = x_offset + (width - 1) / 2 * x_step, y_offset + y_step
      patch = HandlerCrossover.get_patch(cross_x, cross_y)
      ax.add_patch(patch)

      for descendant, x in tmp_coord_dict.items():
        x_coord_dict[descendant] = (x, (cross_x, cross_y))
        color_dict[descendant] = free_colors.pop()
      for c_x, c_y in descendants_coords:
        patch = patch_path(cross_x, cross_y, c_x, c_y)
        ax.add_patch(patch)
      for c_x, c_y in dead_descendants_coords:
        patch = patch_path(cross_x, cross_y, c_x, c_y)
        patch.set_color(unused_path_color)
        ax.add_patch(patch)

      p_x = x_offset + (width - len(parents)) / 2 * x_step
      for parent in parents:
        patch = HandlerIndividual.get_patch(p_x, y_offset)
        patch.set_facecolor(color_dict.get(parent))
        ax.add_patch(patch)
        path = patch_path(p_x, y_offset, cross_x, cross_y)
        ax.add_patch(path)
        p_x += x_step
        pass

      x_offset += width * x_step

    for parent, descendants in mutation_dict.items():
      width = len(descendants)
      p_x, p_y = x_offset + (width - 1) / 2 * x_step, y_offset
      patch = HandlerIndividual.get_patch(p_x, p_y)
      color = color_dict.get(parent)
      patch.set_facecolor(color)
      ax.add_patch(patch)

      mut_x, mut_y = p_x, p_y + y_step
      patch = HandlerMuation.get_patch(mut_x, mut_y)
      ax.add_patch(patch)

      path = patch_path(p_x, p_y, mut_x, mut_y)
      ax.add_patch(path)

      c_x = x_offset
      for descendant in descendants:
        if descendant in genMan_generation:
          x_coord_dict[descendant] = (c_x, (mut_x, mut_y))
          color_dict[descendant] = free_colors.pop()
        else:
          c_y = y_offset + 1.75 * y_step
          patch = HandlerIndividual.get_patch(c_x, c_y)
          ax.add_patch(patch)
          path = patch_path(mut_x, mut_y, c_x, c_y)
          ax.add_patch(path)
        c_x += x_step
      x_offset += width * x_step

    y_offset += height * y_step

    for ind in genMan_generation:
      if ind not in genMan_descendants:
        continue
      color = color_dict.get(ind)
      if color is None:
        color = free_colors.pop()
        color_dict[ind] = color
      gen_acc[ind] = (ind.fitness, color)
      if ind not in x_coord_dict:
        x_, y_ = x_offset, y_offset
        if ind in genMan_selection:
          patch = HandlerSelectedIndividual.get_patch(x_, y_)
          curr_sel_colors[ind] = color
        else:
          patch = HandlerIndividual.get_patch(x_, y_)
        patch.set_facecolor(color)
        ax.add_patch(patch)

        x_offset += x_step
        continue

      x_, (orig_x, orig_y) = x_coord_dict[ind]

      if ind in genMan_selection:
        patch = HandlerSelectedIndividual.get_patch(x_, y_offset)
        curr_sel_colors[ind] = color
      else:
        patch = HandlerIndividual.get_patch(x_, y_offset)

      patch.set_facecolor(color)
      ax.add_patch(patch)

      path = patch_path(orig_x, orig_y, x_, y_offset)
      ax.add_patch(path)

    line_x = x_offset - .25 * x_step
    line_y = y_offset + 1.5 * y_step
    path = patch_path(line_x, y_init_offset - .5 * y_step, line_x, line_y)
    path.set_alpha(.3)
    path.set_color('grey')
    ax.add_patch(path)

    x_offset += .5 * x_step

    for ind in [i for i in genMan_generation if i not in genMan_descendants]:
      color = color_dict.get(ind)
      if color is None:
        color = free_colors.pop()
        color_dict[ind] = color
      sel_acc[ind] = (ind.fitness, color)
      x_, y_ = x_offset, y_offset
      if ind in genMan_selection:
        patch = HandlerSelectedIndividual.get_patch(x_, y_)
        curr_sel_colors[ind] = color
      else:
        patch = HandlerIndividual.get_patch(x_, y_)
      patch.set_facecolor(color)
      ax.add_patch(patch)
      x_offset += x_step

    path = patch_path(11, line_y, x_offset, line_y)
    path.set_alpha(.3)
    path.set_color('grey')
    ax.add_patch(path)

    generation_idx.append(y_offset + y_step)
    generation_label.append('Gen #%i' % (genMan.generation_idx))
    prev_colors = curr_sel_colors
    y_offset += 2 * y_step

    y_ = y_offset - 1 * y_step
    bar_step = -(max(y_init_offset - y_ - .5 * y_step, -3 * y_step)) / (
            len(gen_acc.values()) + len(sel_acc.values()) + 1)
    y_ += .5 * bar_step

    for v, c in sorted(gen_acc.values(), reverse=True):
      y_ -= bar_step
      bars_y.append(y_)
      bars_x.append(v)
      bars_c.append(c)
      bars_h.append(.8 * bar_step)

    y_ -= bar_step
    for v, c in sorted(sel_acc.values(), reverse=True):
      y_ -= bar_step
      bars_y.append(y_)
      bars_x.append(v)
      bars_c.append(c)
      bars_h.append(.8 * bar_step)

    max_x_offset = x_offset if abs(x_offset) > abs(max_x_offset) else max_x_offset

    # WATCHOUT!!
    # magic numbers ahead!
    if genMan.generation_idx % gen_per_plot == 0 and genMan.generation_idx > 0:
      # end plot
      bars_x = np.asarray(bars_x)
      min_v = math.floor(min(bars_x) * 100) / 100 - .01
      bars_x = (bars_x - min_v) / (1 - min_v) * 10
      ax.barh(bars_y, bars_x, height=bars_h, color=bars_c, zorder=2)

      x_lim = [min(-x_step, max_x_offset), max(10, max_x_offset)]
      y_lim = [min(-3 * y_step, y_offset), max(-3 * y_step, y_offset)]
      width = 10 * (x_lim[1] - x_lim[0]) / 25
      height = width * (y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0]) * .81

      plt.rcParams['figure.figsize'] = [width, height]
      fig.set_size_inches(width, height)
      ax.set_ylim(y_lim)

      ax.set_xlabel('Fitness', rotation=-90)
      ax.set_ylabel('')
      y_lim = ax.get_ylim()
      ax.xaxis.set_label_coords(11.5, y_lim[0] - 0.3, ax.transData)

      ax.spines['left'].set_position('zero')
      ax.spines['right'].set_color('none')
      ax.spines['top'].set_color('none')

      xticks = np.linspace(0.0, 10.0, 10, endpoint=True).tolist()
      xlabels = ['{0:.3f}'.format(i) for i in np.linspace(min_v, 1, 10, endpoint=True)]
      plt.xticks(xticks, xlabels, rotation=-90)
      plt.yticks(np.asarray(generation_idx), generation_label)
      ax.grid(True, axis='x')

      ax.set_aspect('equal', anchor='S')
      fig.tight_layout()
      plt.tight_layout()
      plt.savefig(filename + '_%i.pdf' % (plot_idx), format='pdf')
      # plt.savefig(filename + '_%i.svg' % (plot_idx), format='svg')
      plt.close()
      plot_idx += 1

      # Start new plot
      fig = plt.figure()
      ax = fig.gca()
      fontP = mFP()
      fontP.set_size('small')
      plt.legend(legend_elements,
                 ['Individual',
                  'Unused Individual',
                  'Selected Individual',
                  'Crossover',
                  'Mutation'],
                 handler_map={LegendIndividual: HandlerIndividual(),
                              LegendUnusedIndividual: HandlerUnusedIndividual(),
                              LegendSelectedIndividual: HandlerSelectedIndividual(),
                              LegendCrossover: HandlerCrossover(),
                              LegendMutation: HandlerMuation()},
                 ncol=5,
                 loc='upper center',
                 # bbox_to_anchor=(0.5,1.03),
                 prop=fontP,
                 framealpha=1, )

      bars_x = []
      bars_y = []
      bars_c = []
      bars_h = []

      y_offset = 0

      generation_idx = []
      generation_label = []
      max_x_offset = 0

  if y_offset != 0:
    bars_x = np.asarray(bars_x)
    min_v = math.floor(min(bars_x) * 100) / 100
    bars_x = (bars_x - min_v) / (1 - min_v) * 10
    ax.barh(bars_y, bars_x, height=bars_h, color=bars_c, zorder=2)

    x_lim = [min(-x_step, max_x_offset), max(10, max_x_offset)]
    y_lim = [min(-3 * y_step, y_offset), max(-3 * y_step, y_offset)]
    width = 10 * (x_lim[1] - x_lim[0]) / 25
    height = width * (y_lim[1] - y_lim[0]) / (x_lim[1] - x_lim[0]) * .81
    # height = 10
    plt.rcParams['figure.figsize'] = [width, height]
    fig.set_size_inches(width, height)
    ax.set_ylim(y_lim)

    ax.set_xlabel('Fitness', rotation=-90)
    ax.set_ylabel('')
    y_lim = ax.get_ylim()
    ax.xaxis.set_label_coords(11.5, y_lim[0] - 0.3, ax.transData)

    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    xticks = np.linspace(0.0, 10.0, 10, endpoint=True).tolist()
    xlabels = ['{0:.3f}'.format(i) for i in np.linspace(min_v, 1, 10, endpoint=True)]
    plt.xticks(xticks, xlabels, rotation=-90)
    plt.yticks(np.asarray(generation_idx), generation_label)
    ax.grid(True, axis='x')

    ax.set_aspect('equal', anchor='S')
    plt.tight_layout()
    fig.tight_layout()
    plt.savefig(filename + '_%i.pdf' % (plot_idx), format='pdf')
    # plt.savefig(filename + '_%i.svg' % (plot_idx), format='svg')
  plt.close()


def main_gen_hist(prefix, ga_runs):
  for test in ga_runs:
    diversity = sorted(test[1], key=lambda x: x[0])

    sorted_diversity = sorted(diversity, lambda x: x[0])
    # unique_len = list(set(lengths))

    # plot a short run: .35 % of sorted lengths
    run = int(len(sorted_diversity) * .0035)
    generation_history(PlaybackGenManager(filename=sorted_diversity[run][1]),
                       prefix + str(test[0][0]) + '_' + str(test[0][1]) + '_generations_short', 100, NUM_COLORS=18)

    # plot a medium run: middle of sorted lengths
    run = int(len(sorted_diversity) / 2)
    generation_history(PlaybackGenManager(filename=sorted_diversity[run][1]),
                       prefix + str(test[0][0]) + '_' + str(test[0][1]) + '_generations_medium', 100, NUM_COLORS=30)

    # plot longest run: 99.65 % of sorted lengths
    run = int(len(sorted_diversity) * 0.9965)
    generation_history(PlaybackGenManager(filename=sorted_diversity[run][1]),
                       prefix + str(test[0][0]) + '_' + str(test[0][1]) + '_generations_long', 100, NUM_COLORS=30)


# </editor-fold>

# <editor-fold desc="bokeh imports">
from bokeh.plotting import figure as bk_figure, ColumnDataSource as bk_ColumnDataSource
from bokeh.models import Slider as bk_Slider, Range1d as bk_Range1d, Legend as bk_Legend, ColorBar as bk_ColorBar, \
  BasicTicker as bk_BasicTicker
from bokeh.layouts import row as bk_row, widgetbox as bk_widgetbox
from bokeh.server.server import Server as bk_Server
from bokeh.application import Application as bk_Application
from bokeh.application.handlers.function import FunctionHandler as bk_FunctionHandler
from bokeh.models.widgets import Button as bk_Button, Dropdown as bk_Dropdown
from bokeh.transform import linear_cmap as bk_linear_cmap
# </editor-fold>

# Chaco
# add to requirements if used!!
# from chaco.shell import plot

# Surface
from sklearn.neighbors import KNeighborsRegressor

# global svg
svg = False


# <editor-fold desc="prejected search space plot">
def prepare_coords(coords):
  Y, height = zip(*coords.values())
  coords = dict(zip(coords.keys(), Y))
  return coords


def build_tree(ind_dict, ind, parent_dict, sel_list, coords):
  ind_x, ind_y = coords[ind.dna]
  parents = parent_dict.get(ind)
  if parents is None:
    return
  x_, y_ = zip(*[coords[p.dna] for p in parents])
  sym_x, sym_y = np.asarray(x_).sum() / len(parents), np.asarray(y_).sum() / len(parents)
  if len(parents) > 1:
    # Crossover
    ind_dict['x_cro'].append(sym_x)
    ind_dict['y_cro'].append(sym_y)
  else:
    # Mutation
    sym_x, sym_y = (sym_x + ind_x) / 2, (sym_y + ind_y) / 2
    ind_dict['x_mut'].append(sym_x)
    ind_dict['y_mut'].append(sym_y)
  ind_dict['x_lin'].append((sym_x, ind_x))
  ind_dict['y_lin'].append((sym_y, ind_y))
  for parent in parents:
    p_x, p_y = coords[parent.dna]
    ind_dict['x_lin'].append((p_x, sym_x))
    ind_dict['y_lin'].append((p_y, sym_y))
    if parent in sel_list:
      ind_dict['x_par'].append(p_x)
      ind_dict['y_par'].append(p_y)
    else:
      ind_dict['x_tmp'].append(p_x)
      ind_dict['y_tmp'].append(p_y)
      build_tree(ind_dict, parent, parent_dict, sel_list, coords)


def preprocess_generationManager(genMan, coords):
  run_result = []
  for genManager in genMan:
    reverse_dict = {}
    for parents, descendants in genManager.crossover_ancestor:
      for descendant in descendants:
        if descendant in reverse_dict:
          print('already in dict')
        reverse_dict[descendant] = parents
    for parent, descendants in genManager.mutation_ancestor:
      for descendant in descendants:
        if descendant in reverse_dict:
          print('already in dict')
        reverse_dict[descendant] = [parent]

    gen_result = []
    coord_dict = {'x_par': [], 'y_par': [],
                  'x_tmp': [], 'y_tmp': [],
                  'x_mut': [], 'y_mut': [],
                  'x_cro': [], 'y_cro': [],
                  'x_lin': [], 'y_lin': []}
    current_generation = genManager.descendants
    ind_x, ind_y = [], []
    for ind in current_generation:
      ind_dict = {'x_par': [], 'y_par': [],
                  'x_tmp': [], 'y_tmp': [],
                  'x_mut': [], 'y_mut': [],
                  'x_cro': [], 'y_cro': [],
                  'x_lin': [], 'y_lin': []}
      x_, y_ = coords.get(ind.dna)
      ind_x.append(x_)
      ind_y.append(y_)
      ind_dict['x_ind'], ind_dict['y_ind'] = [x_], [y_]
      build_tree(ind_dict, ind, reverse_dict, genManager.prev_selection, coords)

      coord_dict['x_par'], coord_dict['y_par'] = coord_dict['x_par'] + ind_dict['x_par'], \
                                                 coord_dict['y_par'] + ind_dict['y_par']
      gen_result.append(ind_dict)

    coord_dict['x_ind'] = ind_x
    coord_dict['y_ind'] = ind_y

    if len(coord_dict['x_par']) > 1:
      coord_dict['x_par'], coord_dict['y_par'] = zip(*set(zip(coord_dict['x_par'], coord_dict['y_par'])))
    # if len(coord_dict['x_tmp']) > 1:
    #   coord_dict['x_tmp'], coord_dict['y_tmp'] = zip(*set(zip(coord_dict['x_tmp'], coord_dict['y_tmp'])))
    # if len(coord_dict['x_mut']) > 1:
    #   coord_dict['x_mut'], coord_dict['y_mut'] = zip(*set(zip(coord_dict['x_mut'], coord_dict['y_mut'])))
    # if len(coord_dict['x_cro']) > 1:
    #   coord_dict['x_cro'], coord_dict['y_cro'] = zip(*set(zip(coord_dict['x_cro'], coord_dict['y_cro'])))
    # if len(coord_dict['x_lin']) > 1:
    #   coord_dict['x_lin'], coord_dict['y_lin'] = zip(*set(zip(coord_dict['x_lin'], coord_dict['y_lin'])))

    gen_result.insert(0, coord_dict)
    run_result.append(gen_result)
  return run_result


def prepare_data(data, coords):
  coords = prepare_coords(coords=coords)
  result = {}
  for grp, grp_data in data:
    grp_result = []
    for run, run_file in grp_data:
      run_result = preprocess_generationManager(PlaybackGenManager(filename=run_file), coords)
      grp_result.append(run_result)
    result[str(grp[0]) + ', ' + str(grp[1])] = grp_result
  return result


def acc_space_gens(plot_data, coords, show=False, filter_heights=True):
  Y, height = zip(*coords.values())
  # coords = dict(zip(coords.keys(), Y))

  global svg
  print('Prepare data')
  now = datetime.now()
  # plot_data = prepare_data(data=data, coords=coords)
  space_x, space_y = zip(*Y)

  def _distance(distance):
    tmp = np.exp(-distance)
    tmp = tmp / tmp.sum()
    return tmp

  neigh = KNeighborsRegressor(n_neighbors=7, weights=_distance)
  Y = np.asarray(Y)
  height = np.asarray(height)
  if filter_heights:
    Y_, height_ = zip(*[(y, h) for y, h in zip(Y, height) if h >= 0])
  else:
    Y_, height_ = Y, height
  neigh.fit(Y_, height_)
  xmin, xmax = Y[:, 0].min(), Y[:, 0].max()
  ymin, ymax = Y[:, 1].min(), Y[:, 1].max()
  xmin, xmax = xmin - 0.2 * (xmax - xmin), xmax + .2 * (xmax - xmin)
  ymin, ymax = ymin - 0.2 * (ymax - ymin), ymax + .2 * (ymax - ymin)
  # zmin, zmax = height.min(), height.max()
  # scale = max(ymax - ymin, xmax - xmin) / (zmax - zmin)
  x_grid, y_grid = np.meshgrid(np.linspace(xmin, xmax, 1000), np.linspace(ymin, ymax, 1000))
  _shape = x_grid.shape
  t = np.stack([x_grid.reshape((-1)), y_grid.reshape((-1))], axis=-1)
  heights = neigh.predict(t).reshape(_shape)

  contour_lines = ((1 - np.exp(-np.linspace(3, 20, 100))) * max(height)).tolist()
  cl = plt.contour(x_grid, y_grid, heights, contour_lines)
  contour_lines_x, contour_lines_y = [], []
  for c in cl.collections:
    for p in c.get_paths():
      tmp_x, tmp_y = map(list, zip(*p.vertices))
      contour_lines_x.append(tmp_x)
      contour_lines_y.append(tmp_y)
  plt.clf()
  print('Done', datetime.now() - now, '[s]')

  def make_document(doc):
    fig = bk_figure(plot_width=1000, plot_height=1000, output_backend='svg' if svg else 'webgl')

    grp_key = 'grp_data'
    run_key = 'run_data'
    gen_key = 'gen_data'
    ind_key = 'ind_data'
    shared_data = {}

    tmp = list(plot_data.keys())
    tmp = list(zip(tmp, tmp))
    grp_drop_down = bk_Dropdown(label='Select Group', button_type='success',
                                menu=tmp)
    run_slider = bk_Slider(start=0, end=1, value=0, step=1, title='Run')
    gen_slider = bk_Slider(start=0, end=1, value=0, step=1, title='Generation')
    ind_slider = bk_Slider(start=0, end=1, value=0, step=1, title='Individual')
    gen_step_plus = bk_Button(label='Gen +', button_type='success')
    gen_step_minus = bk_Button(label='Gen -', button_type='success')
    shared_data[ind_key] = {'x_ind': [], 'y_ind': [], 'x_par': [], 'y_par': []}
    backend_btn = bk_Button(label='svg' if svg else 'webgl', button_type='success')

    ind_source = bk_ColumnDataSource(data=dict(x_ind=[], y_ind=[]))
    par_source = bk_ColumnDataSource(data=dict(x_par=[], y_par=[]))
    mut_source = bk_ColumnDataSource(data=dict(x_mut=[], y_mut=[]))
    cro_source = bk_ColumnDataSource(data=dict(x_cro=[], y_cro=[]))
    lin_source = bk_ColumnDataSource(data=dict(x_lin=[], y_lin=[]))
    tmp_source = bk_ColumnDataSource(data=dict(x_tmp=[], y_tmp=[]))

    def gen_plus():
      if gen_slider.value < gen_slider.end:
        gen_slider.value += 1

    gen_step_plus.on_click(gen_plus)

    def gen_minus():
      if gen_slider.value > gen_slider.start:
        gen_slider.value -= 1

    gen_step_minus.on_click(gen_minus)

    def grp_change(attrname, old, new):
      grp_data = plot_data[new]
      grp_drop_down.label = new
      shared_data[grp_key] = grp_data

      run_slider.end = len(grp_data) - 1
      run_slider.value = 0

      run_slider.trigger('value', 0, 0)
      pass

    grp_drop_down.on_change('value', grp_change)

    def run_change(attrname, old, new):
      run_data = shared_data[grp_key][new]
      shared_data[run_key] = run_data

      gen_slider.end = len(run_data) - 1
      gen_slider.value = 0

      gen_slider.trigger('value', 0, 0)
      pass

    run_slider.on_change('value', run_change)

    def gen_change(attrname, old, new):
      gen_data = shared_data[run_key][new]
      shared_data[gen_key] = gen_data

      ind_slider.end = len(gen_data) - 1
      ind_slider.value = 0

      ind_slider.trigger('value', 0, 0)
      pass

    gen_slider.on_change('value', gen_change)

    def ind_change(attrname, old, new):
      ind_data = shared_data[gen_key][new]
      shared_data[ind_key] = ind_data
      ind_source.data = dict(x_ind=ind_data.get('x_ind'), y_ind=ind_data.get('y_ind'))
      par_source.data = dict(x_par=ind_data.get('x_par'), y_par=ind_data.get('y_par'))
      mut_source.data = dict(x_mut=ind_data.get('x_mut'), y_mut=ind_data.get('y_mut'))
      cro_source.data = dict(x_cro=ind_data.get('x_cro'), y_cro=ind_data.get('y_cro'))
      lin_source.data = dict(x_lin=ind_data.get('x_lin'), y_lin=ind_data.get('y_lin'))
      tmp_source.data = dict(x_tmp=ind_data.get('x_tmp'), y_tmp=ind_data.get('y_tmp'))
      pass

    ind_slider.on_change('value', ind_change)

    # palette = ['#%06x' % ((255) * 256 * 256 + i * 256 + 0) for i in range(0, 150)] + \
    #           ['#%06x' % ((255 - i) * 256 * 256 + (150 + i) * 256 + 0) for i in range(106)] + \
    #           ['#%06x' % ((255 - i) * 256 * 256 + 255 * 256 + 0) for i in range(150, 256)]

    # palette = ['#%06x' % ((255 - i) * 256 * 256 + i * 256 + 0) for i in range(256)]

    # palette = ['#%06x' % ((255) * 256 * 256 + i * 256 + 0) for i in range(256)] + \
    #           ['#%06x' % ((255-i) * 256 * 256 + 255 * 256 + 0) for i in range(256)]

    # p_ = np.asarray(plt.get_cmap('gist_ncar')(np.linspace(.0,1.,256)))
    p_ = np.asarray(plt.get_cmap('nipy_spectral')(np.linspace(.0, 1., 256)))
    p_ = [colorsys.rgb_to_hsv(i[0], i[1], i[2]) for i in p_]
    p_ = np.asarray([colorsys.hsv_to_rgb(i[0], .8 * i[1], .8 * i[2]) for i in p_])
    p_ = p_ * 255
    palette = ['#%02x%02x%02x' % (int(i[0]), int(i[1]), int(i[2])) for i in p_]

    # fitness heatmap
    fig.image(image=[heights], x=xmin, y=ymin, dw=xmax - xmin, dh=ymax - ymin, palette=palette)
    fig.multi_line(xs=contour_lines_x, ys=contour_lines_y, color='black', line_width=1, alpha=.1)

    # all possible individuals
    space = fig.circle(x=space_x, y=space_y, fill_color='blue', line_color='blue', size=3,
                       alpha=.4)

    # Lines
    fig.multi_line(xs='x_lin', ys='y_lin', source=lin_source, color='black', line_width=2, alpha=.6)

    # individuals of current gen
    individuals = fig.circle(x='x_ind', y='y_ind', source=ind_source, fill_color=(100, 100, 255), line_color='black',
                             size=12)

    # parent individuals / selection of previous gen
    parents = fig.circle(x='x_par', y='y_par', source=par_source, fill_color=(200, 200, 200), line_color='red', size=17,
                         fill_alpha=.3, line_width=2)

    # Intermediate individuals
    fig.square(x='x_tmp', y='y_tmp', source=tmp_source, fill_color=None, fill_alpha=0, line_color='black', size=8)
    intermediate = fig.square([xmin-(xmax-xmin)*.5], [ymin-(ymax-ymin)*.5], fill_color=None, fill_alpha=0, line_color='black', size=8)

    # Crossover
    fig.triangle(x='x_cro', y='y_cro', source=cro_source, fill_color=(138, 69, 0), line_color='black',
                             size=9)
    crossover = fig.triangle([xmin-(xmax-xmin)*.5], [ymin-(ymax-ymin)*.5], fill_color=(138, 69, 0), line_color='black',
                             size=9)

    # Mutation
    fig.hex(x='x_mut', y='y_mut', source=mut_source, fill_color=(38, 38, 110), line_color='black',
                       size=9)
    mutation = fig.hex([xmin-(xmax-xmin)*.5], [ymin-(ymax-ymin)*.5], fill_color=(38, 38, 110), line_color='black',
                       size=9)

    legend = bk_Legend(items=[
      ('Individual', [space]),
      ('Selected indivudal of previous generation', [parents]),
      ('Crossover', [crossover]),
      ('Intermediate individual', [intermediate]),
      ('Mutation', [mutation]),
      ('Individual of current generation', [individuals]),
    ], background_fill_color=None, background_fill_alpha=0)

    mapper = bk_linear_cmap(field_name='Fitness', palette=palette, low=heights.min(),
                            high=heights.max())
    color_bar = bk_ColorBar(color_mapper=mapper['transform'], width=15, location=(0, 0), title='Fitness',
                            label_standoff=8, ticker=bk_BasicTicker(desired_num_ticks=30))

    fig.add_layout(legend)
    fig.add_layout(color_bar, 'right')

    fig.x_range = bk_Range1d(xmin, xmax)
    fig.y_range = bk_Range1d(ymin, ymax)

    stop = bk_Button(label='Stop Server', button_type='success')

    def stop_server():
      print('Stopping Server!')
      stop.label = 'Server Stopped!'
      stop.button_type = 'warning'
      raise SystemExit()

    stop.on_click(stop_server)

    def change_backend():
      global svg
      if fig.output_backend is 'webgl':
        fig.output_backend = 'svg'
        backend_btn.label = 'reload page for svg'
        svg = True
      else:
        fig.output_backend = 'webgl'
        backend_btn.label = 'reload page for webgl'
        svg = False

    backend_btn.on_click(change_backend)

    layout = bk_row(
      fig,
      bk_widgetbox(grp_drop_down, run_slider, gen_slider, gen_step_plus, gen_step_minus, ind_slider, backend_btn,
                   stop),
      width=800
    )

    doc.title = 'Acc Space'
    doc.add_root(layout)

  apps = {'/': bk_Application(bk_FunctionHandler(make_document))}

  server = bk_Server(apps, port=5000)
  if show:
    server.show('/')
  server.run_until_shutdown()
  print('This is not executed when stopping via button!')


# </editor-fold>

# <editor-fold desc="Mayavi imports">
from mayavi.mlab import figure as ma_figure, \
  surf as ma_surf, \
  contour_surf as ma_contour_surf, \
  colorbar as ma_colorbar, \
  axes as ma_axes, \
  show as ma_show


# </editor-fold>


# <editor-fold desc="fitneslandscape plot - work in progress">
def fitness_landscape(coords, filter_heights=True):
  Y, height = zip(*coords.values())
  if filter_heights:
    Y, height = zip(*[(y, h) for y, h in zip(Y, height) if h >= 0])

  def _distance(distance):
    tmp = np.exp(-distance)
    tmp = tmp / tmp.sum()
    return tmp

  neigh = KNeighborsRegressor(n_neighbors=7, weights=_distance)
  Y = np.asarray(Y)
  height = np.asarray(height)
  neigh.fit(Y, height)
  xmin, xmax = Y[:, 0].min(), Y[:, 0].max()
  ymin, ymax = Y[:, 1].min(), Y[:, 1].max()
  xmin, xmax = xmin - 0.2 * (xmax - xmin), xmax + .2 * (xmax - xmin)
  ymin, ymax = ymin - 0.2 * (ymax - ymin), ymax + .2 * (ymax - ymin)
  zmin, zmax = height.min(), height.max()
  scale = max(ymax - ymin, xmax - xmin) / (zmax - zmin)

  # x_grid, y_grid = np.mgrid[Y[:,0].min():Y[:,0].max():0.5j, Y[:,1].min():Y[:,1].max():0.5j]
  y_grid, x_grid = np.meshgrid(np.linspace(xmin, xmax, 200), np.linspace(ymin, ymax, 200))

  def _height(x_grid, y_grid):
    _shape = x_grid.shape
    t = np.stack([x_grid.reshape((-1)), y_grid.reshape((-1))], axis=-1)
    heights = neigh.predict(t).reshape(_shape)
    return heights

  # mlab.title('Search Space Evaluation')
  # f = mlab.figure('Surf', bgcolor=(1,1,1), fgcolor=(0,0,0), size=(1000,1000))
  f = ma_figure("Surf", bgcolor=(1, 1, 1), fgcolor=(0, 0, 0), size=(2000, 2000))
  # print(heights.min(), heights.max())
  s = ma_surf(x_grid, y_grid, _height, warp_scale=scale, colormap='spectral')  # , vmin=min(height), vmax=max(height))
  contour_heights = ((1 - np.exp(-np.linspace(3, 20, 100))) * max(height)).tolist()
  # print(contours)
  cs = ma_contour_surf(x_grid, y_grid, _height, warp_scale=scale, contours=contour_heights, colormap='binary',
                       line_width=1)
  s.actor.actor.scale = (1, 1, 1)
  s.actor.property.lighting = False
  ma_colorbar(s, title='Fitness', orientation='vertical')

  # mlab.options.offscreen = True
  ma_axes(s, x_axis_visibility=True, xlabel='', line_width=4,
          y_axis_visibility=True, ylabel='',
          z_axis_visibility=True, zlabel='',
          ranges=(-0.5, .5, -0.5, .5, min(height), max(height)))
  # f.scene.isometric_view()
  ma_show()

# </editor-fold>
