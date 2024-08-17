## import python data analysis library
import numpy as np
import pandas as pd

## import data visualization library matplotlib and seaborn
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator, FuncFormatter
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FuncFormatter
from matplotlib import rc
from mpl_toolkits.axes_grid.inset_locator import (inset_axes, InsetPosition, mark_inset, zoomed_inset_axes)

## import module to read files
import mimetypes
import urllib
import os

# import google colab to use google colab as editor
# if you use other editor, do not need to import google.colab here
#from google.colab import drive
#drive.mount('/content/drive')

## setting path
# get current path
path_current = os.getcwd()
path_current=path_current.replace('\\', '/')
# the path is where the dataset saved
path = path_current + '/Example_Data/Line/'

# the "path_img" is the position where final image will be saved
path_img = path_current + '/Images/'

class line:
    def __init__(self,path=path,path_img=path_img):
        self.path=path
        self.path_img=path_img
        ## Configuration of the line chart
        # plotwidth: width of the plot
        # plotheight: height of the plot
        # backgrid: backgrid of the plot
        # isframe: frame of the plot
        # my_font: the typeface of x, y labels
        # linewidth: linewidth of the lines in the plot
        # gridlinewidth: if backgrid is True, grid linewidth is the line width of background grid
        # sa_linecolor: single column paper linecolors palette
        # da_linecolor: double columns paper linecolors palette
        # linestyle: line shapes library
        # labeltext_size: text size of x,y labels
        # labelpad: pad size of label
        # legend_size: size of legend
        # legend_loc: location of legend
        # ncol: number of columns of legend
        # title: True or False as options. If it is True, add title for the plot
        # title_pad: if the title is True, modify pad size of title
        # title_size: if the title is True, modify size of title
        # title_loc: if the title is True, modify location of title
        # markers: True or False as options. If it is True, add markers
        # markersize: if the title is True, modify size of marker
        # markers_shape: shapes library of marker
        # xy_lim: True or False as options. If it is True, add x and y axis' value range
        # x_range: if the xy_lim is True, set x axis range
        # y_range: if the xy_lim is True, set x axis range
        # inset: True or False as options. If it is True, add inset plot
        # xin_start: if the inset is True, the inset plot x axis starts from xin_start
        # xin_end: if the inset is True, the inset plot x axis ends from xin_end
        # yin_start: if the inset is True, the inset plot y axis starts from yin_start
        # yin_end:  if the inset is True, the inset plot y axis starts from yin_end
        # ticks: True or False as options. If it is True, add ticks of x and y axis.
        # tick_size: size of tick
        # tick_direction: 'out', 'inout' and 'in' options.
        # x_minor_locator: number of minor ticks in x axis
        # y_minor_locator: number of minor ticks in y axis
        # present_linevalue: True or False as options. If it is True, present point value in the line
        # double_axis: True or False as options. If it is True, add second axis on the right of figure
        # save_image: True or False as options. If it is True, save chart
        # savefig_bbox_inches: Bounding box in inches
        self.conf={'plotwidth':9,
                   'plothight':6,
                   'backgrid':True,
                   'isframe':True,
                   'my_font':'DejaVu Sans',
                   'linewidth':2,
                   'gridlinewidth':0.5,
                   'sa_linecolor':['#0173B2', '#DE8F05', '#029E73', '#D55E00',
                                   '#CC78BC', '#8E5638', '#FBAFE4', '#949494',
                                   '#ECE133', '#56B4E9'],
                   'da_linecolor':{'left':['#D6DEBF','#AECEA1','#82BB92','#5EA28D',
                                           '#49838A','#3E5F7E','#383C65','#2B1E3E'],
                                   'right':['#2C1E3D','#51315E','#764476','#9A5B88',
                                            '#B77495','#CF91A3','#E0B1B4','#EDD1CB']},
                   'bg_color':["#ffd6a5","#fdffb6","#99d98c","#bde0fe",'#ffadad',"#48bfe3"],
                   'linestyle':['-',':','-.','--','-',':','-.','--','-','-',':','-.'],
                   'labeltext_size':18,
                   'labelpad':10,
                   'legend_size':10,
                   'legend_loc':'upper right',
                   'ncol':2,
                   'title':False,
                   'title_pad':10,
                   'title_size':20,
                   'title_loc':'center',
                   'markers':False,
                   'markersize':8,
                   'markers_shape':['o','v','D','X','P','2','p','x','d','4','<','*'],
                   'xy_lim':True,
                   'x_range':False,
                   'y_range':False,
                   'xin_start':0,
                   'xin_end':0,
                   'yin_start':0,
                   'yin_end':0,
                   'ticks':True,
                   'tick_size':14,
                   'tick_direction':'out',
                   'x_minor_locator':5,
                   'y_minor_locator':2,
                   'present_linevalue':False,
                   'double_axis':False,
                   'shadow':False,
                   'inset':False,
                   'save_image':False,
                   'savefig_bbox_inches':'tight'}


    ## read file function: read three kinds of format file csv/excel/text
    # if you have other format of file, please change the function manually
    # file: str, filename (e.g.'Vertical_Bar.txt')
    def read_file(self,file):
        file_url = urllib.request.pathname2url(file)
        ftype = mimetypes.guess_type(file_url, strict=True)[0]

        ## read data file according to its formate, default includes three types of files: csv/excel/text
        # read csv format data from the parking dataset

        if 'csv'or 'excel' in ftype:
            # usecols: return a subset of the columns, here choose one column to use in the line chart
            data = pd.read_csv(self.path+file)
        # read excel format data from the parking dataset
        elif 'sheet' in ftype:
            data = pd.read_excel(self.path+file)
        # read text format data from the parking dataset
        elif ftype == 'text/plain':
            data = pd.read_csv(self.path+file, sep="\t")
        else:
            print("File type cannot find!")
        return data

    ## line chart plotting function
    # file: file name of your data source
    # x_col_name: ['index'] or ['x_column_name_a','x_column_name_b'...]
    # y_col_name: ['y_column_name_a','y_column_name_b'...]
    # x_label: x axis label
    # y_label: y axis label
    # legend_label: legend labels names
    # paper_type: 'single' or 'double'
    def plot(self, file, x_col_name=None, y_col_name=None,legend_label=None, x_label=None, y_label=None, **kwargs):
        # read file
        try:
            self.data = self.read_file(file)
        except Exception:
            print('Sorry, this file does not exist, please check the file name')


        cols = self.data.columns.to_list()
        if not x_col_name:
            if 'X' in cols:
                self.x_col_name = ['X']
            else:
                self.x_col_name = ['index']
        else:
            self.x_col_name = x_col_name

        if not y_col_name:
            self.y_col_name = []
            if 'X' in cols:
                cols.remove('X')
            for col in cols:
                if "Unnamed" in col:
                    continue
                self.y_col_name.append(col)
        else:
            self.y_col_name = y_col_name

        if not legend_label:
            self.legend_label = self.y_col_name
        else:
            self.legend_label = legend_label

        if not x_label:
            self.x_label = 'X'
        else:
            self.x_label = x_label


        if not y_label:
            if self.conf['double_axis'] == False:
                self.y_label = 'Y'
            elif self.conf['double_axis'] == True:
                self.y_label = ['Y1', 'Y2']
        else:
            self.y_label = y_label

        # when new configuraton is set, update the original one
        self.conf.update(kwargs)

        cols = self.data.columns.to_list()
        if self.conf['double_axis'] == False:
            for x_col in self.x_col_name:
                if x_col != 'index' and x_col not in cols:
                    raise KeyError("Invaild x_col_name")
            if self.conf['shadow'] == True:
                for y_cols in self.y_col_name:
                    for y_col in y_cols:
                        if y_col not in cols:
                            raise KeyError("Invaild y_col_name")
            else:
                for y_col in self.y_col_name:
                    if y_col not in cols:
                        raise KeyError("Invaild y_col_name")
        elif self.conf['double_axis'] == True:
            for x_col in self.x_col_name[0]:
                if x_col != 'index' and x_col not in cols:
                    raise KeyError("Invaild x_col_name")
            for x_col in self.x_col_name[1]:
                if x_col != 'index' and x_col not in cols:
                    raise KeyError("Invaild x_col_name")
            for y_col in self.y_col_name[0]:
                if y_col not in cols:
                    raise KeyError("Invaild y_col_name")
            for y_col in self.y_col_name[1]:
                if y_col not in cols:
                    raise KeyError("Invaild y_col_name")



        self.len_sac = len(self.conf['sa_linecolor'])
        self.len_dac_l = len(self.conf['da_linecolor']['left'])
        self.len_dac_r = len(self.conf['da_linecolor']['right'])
        self.len_lsty = len(self.conf['linestyle'])
        self.len_mksh = len(self.conf['markers_shape'])


        #Initializing the line
        ## plot size setting
        # figsize: the size of the line chart, (width,hight)
        fig, self.ax_1eft = plt.subplots(figsize =(self.conf['plotwidth'], self.conf['plothight']))

        ## background grid setting
        if self.conf['backgrid'] == True:
            self.ax_1eft.grid(linestyle="--", linewidth=self.conf['gridlinewidth'], color='gray', alpha=0.5)


        if self.conf['double_axis'] == False:
            if self.conf['shadow'] == True:
                self.plot_shadow()
            else:
                self.plot_single_axis()
        else:
            self.plot_double_axis()

        ## x,y axis range limitation
        if self.conf['x_range'] == False:
            pass
        else:
            # x axis valure start from xaxis_start value, and end in max of the file add a small range of value
            self.ax_1eft.set_xlim(self.conf['x_range'][0], self.conf['x_range'][1])

        if self.conf['y_range'] == False:
            pass
        else:
            # y axis valure start from yaxis_start value, and end in max of the file add a small range of value
            self.ax_1eft.set_ylim(self.conf['y_range'][0], self.conf['y_range'][1])

        ## ticks setting
        if self.conf['ticks'] == True:
            ## major and minor ticks for x axia
            # set x axis' detailed ticks label
            self.ax_1eft.xaxis.set_minor_locator(AutoMinorLocator(self.conf['x_minor_locator']))
            # set y axis' detailed ticks label
            self.ax_1eft.yaxis.set_minor_locator(AutoMinorLocator(self.conf['y_minor_locator']))
        elif self.conf['ticks'] == False:
            pass

        ## size of numbers on the ticks of x,y axis' setting
        for tick in self.ax_1eft.xaxis.get_major_ticks():
            tick.label.set_fontsize(self.conf['tick_size'])
        for tick in self.ax_1eft.yaxis.get_major_ticks():
            tick.label.set_fontsize(self.conf['tick_size'])



        # if False, top and right borders removing
        if self.conf['isframe'] == False:
            self.ax_1eft.spines['top'].set_visible(False)
            self.ax_1eft.spines['right'].set_visible(False)

        ## set x, y tick's direction, default:out, can be set to in,out,inout
        if self.conf['tick_direction'] == 'in':
            matplotlib.rcParams['xtick.direction'] = 'in'
            matplotlib.rcParams['ytick.direction'] = 'in'
        elif self.conf['tick_direction'] == 'inout':
            matplotlib.rcParams['xtick.direction'] = 'inout'
            matplotlib.rcParams['ytick.direction'] = 'inout'
        elif self.conf['tick_direction'] == 'out':
            matplotlib.rcParams['xtick.direction'] = 'out'
            matplotlib.rcParams['ytick.direction'] = 'out'

        ## legend setting
        # ncol: number of legend column
        # loc: position of the legend
        if self.conf['double_axis'] == True:
            self.ax_1eft.legend(ncol=self.conf['ncol'], loc=self.conf['legend_loc_l'], fontsize=self.conf['legend_size'])
            self.ax_right.legend(ncol=self.conf['ncol'], loc=self.conf['legend_loc_r'], fontsize=self.conf['legend_size'])
        else:
            self.ax_1eft.legend(ncol=self.conf['ncol'], loc=self.conf['legend_loc'], fontsize=self.conf['legend_size'])


        ## title and position
        if self.conf['title'] == False:
            pass
        else:
            self.ax_1eft.set_title(self.conf['title'], fontsize=self.conf['title_size'], loc=self.conf['title_loc'], pad=self.conf['title_pad'])

        ## save image as pdf to path folder
        # bbox in inches, only the given portion of the figure is saved, figure out the tight bbox of the figure
        if self.conf['save_image'] == True:
            plt.savefig(self.path_img+'line_chart.pdf', bbox_inches=self.conf['savefig_bbox_inches'])

        # showing the image
        plt.show()

    def plot_func(self, ax, x_col_name, y_col_name, legend_label, cur_color, len_color):
        if x_col_name[0] == 'index':
            # line with markers
            if self.conf['markers'] == True:
                for i in range(0, len(y_col_name)):
                    ax.plot(self.data.index, self.data[y_col_name[i]], marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                            linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=legend_label[i], color=cur_color[i%len_color])
            # line without markers
            else:
                for i in range(0, len(y_col_name)):
                    ax.plot(self.data.index, self.data[y_col_name[i]], linestyle=self.conf['linestyle'][0],
                            linewidth=self.conf['linewidth'], label=legend_label[i], color=cur_color[i%len_color])
        # x column is not index
        else:
            # line with markers
            if self.conf['markers'] == True:
                for i in range(0, len(y_col_name)):
                    if len(x_col_name)==1:
                        ax.plot(self.data[x_col_name[0]], self.data[y_col_name[i]], marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                                linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=legend_label[i], color=cur_color[i%len_color])
                    else:
                        ax.plot(self.data[x_col_name[i]], self.data[y_col_name[i]], marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                                linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=legend_label[i], color=cur_color[i%len_color])
            # line without markers
            else:
                for i in range(0, len(y_col_name)):
                    if len(x_col_name)==1:
                        ax.plot(self.data[x_col_name[0]], self.data[y_col_name[i]], linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'],
                                label=legend_label[i], color=cur_color[i%len_color])
                    else:
                        ax.plot(self.data[x_col_name[i]], self.data[y_col_name[i]], linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'],
                                label=legend_label[i], color=cur_color[i%len_color])



    def plot_single_axis(self):
        self.ax_1eft.set_xlabel(self.x_label, fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])
        self.ax_1eft.set_ylabel(self.y_label, fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])

        self.plot_func(self.ax_1eft, self.x_col_name, self.y_col_name, self.legend_label, self.conf['sa_linecolor'], self.len_sac)

        if self.conf['inset'] == True:
            self.plot_single_inset()

        ## presenting values on graph
        if self.conf['present_linevalue'] == True:
            if self.x_col_name[0] == 'index':
                for i in range(0, len(self.y_col_name)):
                    for x,y in zip(self.data.index.values.tolist(),self.data[self.y_col_name[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])

            else:
                for i in range(0, len(self.y_col_name)):
                    for x,y in zip(self.data[self.x_col_name[i]].values.tolist(),self.data[self.y_col_name[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])
        else:
            pass

    def plot_single_inset(self):
        axins = zoomed_inset_axes(self.ax_1eft, 3, bbox_to_anchor=(0.43,0.7), bbox_transform=self.ax_1eft.transAxes)

        ## drawing lines of inset plot
        # use the plot function
        # linewidth: the line width, here set to 2
        # x column is index
        self.plot_func(axins, self.x_col_name, self.y_col_name, self.legend_label, self.conf['sa_linecolor'], self.len_sac)

        ## inset plot x,y axis range limit
        axins.set_xlim(0.5, 0.75)
        axins.set_ylim(0, 5)

        mark_inset(self.ax_1eft, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    # double axis line chart
    def plot_double_axis(self):
        ## x, y axis setting
        # fontsize: x, y title size
        # labelpad: scalar, optional, default: None
        self.ax_1eft.set_xlabel(self.x_label, fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])
        self.ax_1eft.set_ylabel(self.y_label[0], fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])


        # left x,y columns and labels
        self.x_col_name_l = self.x_col_name[0]
        self.y_col_name_l = self.y_col_name[0]
        self.legend_label_l = self.legend_label[0]

        # use the plot function
        self.plot_func(self.ax_1eft, self.x_col_name_l, self.y_col_name_l, self.legend_label_l, self.conf['da_linecolor']['left'], self.len_dac_l)

        ## instantiate a left axis that shares the same x-axis
        self.ax_right = self.ax_1eft.twinx()

        self.ax_right.set_ylabel(self.y_label[1], fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])

        # already handled the x-label with ax_1eft
        # alpha: transparency, soft color
        # right x,y columns and labels
        self.x_col_name_r = self.x_col_name[1]
        self.y_col_name_r = self.y_col_name[1]
        self.legend_label_r = self.legend_label[1]

        # use the plot function
        self.plot_func(self.ax_right, self.x_col_name_r, self.y_col_name_r, self.legend_label_r, self.conf['da_linecolor']['right'], self.len_dac_r)


        # plotting inset plot
        if self.conf['inset'] == True:
            self.plot_double_inset()

        ## presenting left_axis values on graph
        if self.conf['present_linevalue'] == True:
            if self.x_col_name_l[0] == 'index':
                for i in range(0, len(self.y_col_name_l)):
                    for x,y in zip(self.data.index.values.tolist(),self.data[self.y_col_name_l[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])

            else:
                for i in range(0, len(self.y_col_name_l)):
                    for x,y in zip(self.data[self.x_col_name_l[i]].values.tolist(),self.data[self.y_col_name_l[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])
        else:
            pass

        ## presenting right_axis values on graph
        if self.conf['present_linevalue'] == True:
            if self.x_col_name_r[0] == 'index':
                for i in range(0, len(self.y_col_name_r)):
                    for x,y in zip(self.data.index.values.tolist(),self.data[self.y_col_name_r[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])

            else:
                for i in range(0, len(self.y_col_name_r)):
                    for x,y in zip(self.data[self.x_col_name_r[i]].values.tolist(),self.data[self.y_col_name_r[i]].values.tolist()):
                        plt.text(x, y+0.01, str(round(y,2)), color = self.color[i])
        else:
            pass


    def plot_double_inset(self):
        zoomed_left_axis = zoomed_inset_axes(self.ax_1eft, 1, bbox_to_anchor=(0.43,0.7), bbox_transform=self.ax_1eft.transAxes)

        ## drawing lines of inset plot
        # use the plot function
        # linewidth: the line width, here set to 2
        # x column is index
        self.plot_func(zoomed_left_axis, self.x_col_name_l, self.y_col_name_l, self.legend_label_l, self.conf['da_linecolor']['left'], self.len_dac_l)


        ## inset plot x,y axis range limit
        zoomed_left_axis.set_xlim(self.conf['xin_start'], self.conf['xin_end'])
        zoomed_left_axis.set_ylim(self.conf['yin_start'], self.conf['yin_end'])

        zoomed_right_axis = zoomed_left_axis.twinx()

        ## drawing lines of inset plot
        # use the plot function
        # linewidth: the line width, here set to 2
        # x column is index
        self.plot_func(zoomed_right_axis, self.x_col_name_r, self.y_col_name_r, self.legend_label_r, self.conf['da_linecolor']['right'], self.len_dac_r)


    def plot_shadow(self):
        self.ax_1eft.set_xlabel(self.x_label, fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])
        self.ax_1eft.set_ylabel(self.y_label, fontproperties=self.conf['my_font'], fontsize=self.conf['labeltext_size'], labelpad=self.conf['labelpad'])

        y_data = []
        for i in range(len(self.y_col_name)):
            y_tmp = []
            for j in range(len(self.y_col_name[i])):
                y_tmp.append(self.data[self.y_col_name[i][j]])
            y_data.append(y_tmp)

        if self.x_col_name[0] == 'index':
            # line with markers
            if self.conf['markers'] == True:
                for i in range(0, len(y_data)):
                    self.ax_1eft.plot(self.data.index, np.mean(y_data[i], axis=0), marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                                      linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                    self.ax_1eft.fill_between(self.data.index, np.max(y_data[i], axis=0),np.min(y_data[i], axis=0), facecolor=self.conf['bg_color'][i], alpha=0.45)
            # line without markers
            else:
                for i in range(0, len(y_data)):
                    self.ax_1eft.plot(self.data.index, np.mean(y_data[i], axis=0), linestyle=self.conf['linestyle'][0],
                                      linewidth=self.conf['linewidth'], label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                    self.ax_1eft.fill_between(self.data.index, np.max(y_data[i], axis=0),np.min(y_data[i], axis=0), facecolor=self.conf['bg_color'][i], alpha=0.45)

        # x column is not index
        else:
            # line with markers
            if self.conf['markers'] == True:
                for i in range(0, len(y_data)):
                    if len(self.x_col_name)==1:
                        self.ax_1eft.plot(self.data[self.x_col_name[0]], np.mean(y_data[i], axis=0), marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                                          linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                        self.ax_1eft.fill_between(self.data[self.x_col_name[0]], np.max(y_data[i], axis=0),np.min(y_data[i], axis=0), facecolor=self.conf['bg_color'][i], alpha=0.45)
                    else:
                        self.ax_1eft.plot(self.data[self.x_col_name[i]], np.mean(y_data[i], axis=0), marker=self.conf['markers_shape'][i%self.len_mksh], markersize=self.conf['markersize'],
                                          linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'], label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                        self.ax_1eft.fill_between(self.data[self.x_col_name[i]], np.max(y_data[i], axis=0),np.min(y_data[i], axis=0), facecolor=self.conf['bg_color'][i], alpha=0.45)
            # line without markers
            else:
                for i in range(0, len(y_data)):
                    if len(self.x_col_name)==1:
                        self.ax_1eft.plot(self.data[self.x_col_name[0]], np.mean(y_data[i], axis=0), linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'],
                                          label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                        self.ax_1eft.fill_between(self.data[self.x_col_name[0]], np.max(y_data[i], axis=0),np.min(y_data[i], axis=0), facecolor=self.conf['bg_color'][i], alpha=0.45)
                    else:
                        self.ax_1eft.plot(self.data[self.x_col_name[i]], np.mean(y_data[i], axis=0), linestyle=self.conf['linestyle'][0], linewidth=self.conf['linewidth'],
                                          label=self.legend_label[i], color=self.conf['sa_linecolor'][i%self.len_sac])
                        self.ax_1eft.fill_between(self.data[self.x_col_name[i]], np.max(y_data[i], axis=0),np.min(y_data[i], axis=0),facecolor=self.conf['bg_color'][i],alpha=0.45)
