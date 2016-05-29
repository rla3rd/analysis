#!/usr/bin/python
# import modules
import os
import tempfile
import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import  IndexLocator, FuncFormatter, NullFormatter, MultipleLocator, NullLocator, FormatStrFormatter
from matplotlib.dates import IndexDateFormatter, date2num, DateFormatter, MonthLocator, WeekdayLocator
import numpy as np
from PIL import Image
from PIL import ImageEnhance
import math

# import custom modules
from colors import RGBColors

class Chart():
    """
    Plot charts class
    """
    #initialize class variables
    def __init__(self, dimensions=(12,8)):
        self.size = None
        self.draw = RGBColors()
        self.figureBG = self.draw.color['White']
        self.axesBG = self.draw.color['White']
        self.bearish = '#ffc1c1'
        self.bullish = '#bced91'
        self.textsize = 10
        self.dpi = 100
        self.figure = plt.figure(1, facecolor=self.figureBG, figsize=dimensions, dpi=self.dpi)
	self.figure.set_frameon(False)
        self.nullfmt = NullFormatter()
        self.rgbcolor = '0.75'
        self.rcParams = plt.rcParams
        self.rcParams['grid.color'] = self.rgbcolor
        self.rcParams['grid.linestyle'] = '-'
        self.rcParams['grid.linewidth'] = 0.5
        self.rcParams['xtick.labelsize'] = self.textsize
        self.rcParams['ytick.labelsize'] = self.textsize
        self.rcParams['font.weight'] = 'semibold'
        self.left=0.08
        self.width = 0.88
        self.interval = None
        self.canvas = plt.FigureCanvasBase( self.figure )
        self.bgcolor = self.draw.color['White'] 
        self.dpi = 100
        self.majorxticks = MonthLocator(interval=3)
        self.minorxticks = MonthLocator(interval=1)
        self.monthFmt = ('%b %y')
        self.dateFmt = ('%b-%d %y')

    def save(self, filename, watermark=None,):
        self.tempfilenum, self.tempfilename=tempfile.mkstemp(suffix='.png')
        if watermark == None:
            self.canvas.print_figure(filename, facecolor=self.bgcolor, edgecolor=self.bgcolor, dpi=self.dpi )
        else:
            tempfilenum, tempfilename=tempfile.mkstemp(suffix='.png')
            self.canvas.print_figure( tempfilename, facecolor=self.bgcolor, edgecolor=self.bgcolor, dpi=self.dpi )
            imagefile=Image.open(tempfilename)
            #open the watermark file according to size get variable
            wmfile = Image.open(watermark)
            #merge temp and watermark files for display
            mergefile = self.watermark(imagefile, wmfile, 'scale', 0.75)
            os.close(tempfilenum)
            os.remove(tempfilename)
            mergefile.save(filename)

    def get_locator(self, days):
        """
        the axes cannot share the same locator, so this is a helper
        function to generate locators that have identical functionality
        """
        return MonthLocator(interval=1)
 
    def qtr2string(self, qtr):
        """
        converts year and quarter values YYYY.MM to string values for use as
        a dictionary key
        """
        
        yr = qtr[0:4]
        qtr = qtr[5:]
        return "Q%s\'%s" % (qtr, yr[2:])
    
    def plotFill(self, axName, x, var1, var2, color, edge=RGBColors().color['Gray'], a=0.5):
        """ 
        shades in the space between 2 variables with the specified color
        """

        x1 = x
        y1 = var1
        y2 = var2

        x = np.concatenate((x1, x1[::-1]))
        y = np.concatenate((y1, y2[::-1]))
        self.subplot[axName].fill(x, y, facecolor=color, edgecolor=edge, alpha=a)

    def setAxes(self, axesCount):
        """
        predefined axes locations for a given axis count
        """
        self.rects = {}
        if axesCount == 1:
            self.rects['axMain'] = [self.left, 0.12, self.width, 0.86]
        elif axesCount == 2:
            self.rects['axMain'] = [self.left, 0.50, self.width, 0.44]
            self.rects['axSub1'] = [self.left, 0.15, self.width, 0.25]
        elif axesCount == 3:
            self.rects['axMain'] = [self.left, 0.6, self.width, 0.35]
            self.rects['axSub1'] = [self.left, 0.37, self.width, 0.18]
            self.rects['axSub2']= [self.left, 0.14, self.width, 0.18]
        elif axesCount == 4:
            self.rects['axMain'] = [self.left, 0.7, self.width, 0.25]
            self.rects['axSub1'] = [self.left, 0.55, self.width, 0.15]
            self.rects['axSub2'] = [self.left, 0.32, self.width, 0.15]
            self.rects['axSub3'] = [self.left, 0.12, self.width, 0.15]

        self.subplot = {}
        for key in self.rects.keys():
            self.subplot[key] = self.setPlotProperties(self.rects[key])

    def setPlotProperties(self, rect, locator=True, days=20):
        """
        sets general plot properties for a given axis
        """
	self.rcParams['font.family'] = 'sans-serif'
       	self.rcParams['axes.edgecolor'] = 'gray' 
        plotaxis = plt.axes(rect, axisbg=self.axesBG)
        plotaxis.set_axis_bgcolor(self.axesBG)
        plotaxis.yaxis.tick_left()
        plotaxis.xaxis.set_major_locator(self.get_locator(days))
        if locator:
            plotaxis.yaxis.set_major_locator(MultipleLocator(5))
        plotaxis.grid(True)
        plotaxis.set_axisbelow( True )
        plotaxis.xaxis.set_major_formatter(self.nullfmt)
        return plotaxis

    def clone(self, axName, scaled=True):
        """
        sets a cloned axis on top of the existing one with the ticks on the right
        """

        self.subplot['%sRight' % axName] = self.subplot[axName].twinx()
        self.subplot['%sRight' % axName].set_ylim(self.subplot[axName].get_ylim())
        self.subplot['%sRight' % axName].set_xlim(self.subplot[axName].get_xlim())
        if scaled:
            self.subplot['%sRight' % axName].set_yscale(self.subplot[axName].get_yscale())
        self.subplot['%sRight' % axName].xaxis.set_major_locator(self.majorxticks)
        self.subplot['%sRight' % axName].xaxis.set_minor_locator(self.minorxticks)
        self.subplot['%sRight' % axName].xaxis.set_major_formatter(DateFormatter(self.dateFmt))

    def setAxisTitle(self, axName, title, pos='above', textsize=10):
        """
        set the title position and size
        """
        title = self.subplot[axName].set_title(title, fontsize=textsize)
        if pos=='above':
            title.set_y(1.05)
            title.set_x(0)
        elif pos=='above2':
            title.set_y(1.1)
            title.set_x(0)
        elif pos=='below':
            title.set_y(0.8)
            title.set_x(0.01)
        elif pos=='below2':
            title.set_y(0.75)
            title.set_x(0.01)
        title.set_horizontalalignment('left')

    def setChartTitle(self, title, textsize=12):
        self.title = plt.title(title, fontsize=textsize)
        self.title.set_y(1.05)

    def getTicks(self, ticknum, lowerlimit, upperlimit):
        if (upperlimit - lowerlimit) >= 5:
            ticks=round((upperlimit-lowerlimit)/ticknum, 0)
        elif (upperlimit - lowerlimit) >= 1:
            ticks = (upperlimit-lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.05)
            ticks = round(ticks, 2)
        elif (upperlimit - lowerlimit) >= 1 and (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.09:
            ticks=round((upperlimit-lowerlimit)/ticknum, 1)
        elif (upperlimit - lowerlimit) >= 1 and (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.01:
            ticks=round((upperlimit-lowerlimit)/ticknum, 2)
        elif (upperlimit - lowerlimit) >= 1 and (upperlimit - lowerlimit) > 0 and lowerlimit > 0.00:
            ticks=round((upperlimit-lowerlimit)/ticknum, 3)
        elif (upperlimit - lowerlimit) < 1 and (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.09:
            ticks = (upperlimit-lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.1)
            ticks = round(ticks, 1)
        elif (upperlimit - lowerlimit) < 1 and (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.01:
            ticks = (upperlimit-lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.005)
            ticks = round(ticks, 2)
        elif (upperlimit - lowerlimit) < 1 and (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.0:
            ticks = (upperlimit-lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.005)
            ticks = round(ticks, 3)
        elif (upperlimit - lowerlimit) < 1 and (upperlimit - lowerlimit) > 0 and lowerlimit < 0.0:
            if abs(upperlimit) > abs(lowerlimit):
                ticks = abs(upperlimit)/ticknum
            else:
                ticks = abs(lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.005)
            ticks = round(ticks, 3)
        else:
            ticks = (upperlimit-lowerlimit)/ticknum
            ticks = ticks - (ticks % 0.05)
            ticks = round(ticks, 2)

        tickList=[]
        ticklabelList=[]
        if (upperlimit - lowerlimit) >= 5:
            tick = math.ceil(lowerlimit)
        elif (upperlimit - lowerlimit) >= 1:
            tick=lowerlimit - (lowerlimit % 0.05)
            tick=round(tick, 2)
        elif (upperlimit - lowerlimit) > 0 and lowerlimit >= 0.05:
            tick=lowerlimit - (lowerlimit % 0.05)
            tick=round(tick, 2)
        elif (upperlimit - lowerlimit) > 0 and lowerlimit > 0.00:
            tick=lowerlimit - (lowerlimit % 0.005)
            tick=round(tick, 3)
        else:
            tick = math.ceil(lowerlimit)
            tick=round(tick, 2)
        tickList.append(tick)
        ticklabelList.append(str(tick))
        while len(tickList)<=ticknum:
            tick += ticks
            tickList.append(tick)
            ticklabelList.append(str(tick))
        for tick in tickList:
            if tick <= 0:
                tickList.remove(tick)
                ticklabelList.remove(str(tick))
        return (tickList, ticklabelList)

    def plotLog(self, axName, xvar, yvar, ticknum= 5, limits = None, color = 'blue', width=1, offset=0.97):
        """
        plotting function for plotting on a logarithmic scale
        """
        if limits == None:
            if np.isnan(min(yvar)) or np.isnan(max(yvar)):
                lowerlimit = yvar.min() * 0.95
                upperlimit = yvar.max() * 1.05
            else:
                lowerlimit = min(yvar) * 0.95
                upperlimit = max(yvar) * 1.05
        else:
            lowerlimit = min(limits)
            upperlimit = max(limits)
        ( ticks, tickLabels ) = self.getTicks(ticknum, lowerlimit, upperlimit)
        self.subplot[axName].plot( xvar, yvar, color=color, linewidth=width)
        self.subplot[axName].set_yscale('log')
        self.subplot[axName].set_yticks(ticks)
        self.subplot[axName].set_yticklabels(tickLabels)
        self.subplot[axName].set_ylim(lowerlimit, upperlimit)
        for label in self.subplot[axName].get_xticklabels():
            label.set_visible(False)

    def overlay(self, axName, xvar, yvar, ticknum = 5, limits = None, color = 'red', marker='o', offset=1, size = 50):
        """
        plotting function for overlaying a scatter plot over top of an existing plot
        """
        if limits == None:
            if np.isnan(min(yvar)) or np.isnan(max(yvar)):
                lowerlimit = yvar.min() * 0.95
                upperlimit = yvar.max() * 1.05
            else:
                lowerlimit = min(yvar) * 0.95
                upperlimit = max(yvar) * 1.05
        else:
            lowerlimit = min(limits)
            upperlimit = max(limits)

        self.subplot[axName].scatter(xvar, yvar, s = size, marker=marker, c=color)
        self.subplot[axName].set_ylim(lowerlimit, upperlimit)

    def plotLine(self, axName, xvar, yvar, ticknum=5, limits = None, color='blue', flip=False, width=1):
        """ 
        plotting function which draws a line along the supplied x, y coordinates
        """
        if limits == None:
            if np.isnan(min(yvar)) or np.isnan(max(yvar)):
                lowerlimit = yvar.min() * 0.95
                upperlimit = yvar.max() * 1.05
            else:
                lowerlimit = min(yvar) * 0.95
                upperlimit = max(yvar) * 1.05
        else:
            lowerlimit = min(limits)
            upperlimit = max(limits)
        
        (ticks, tickLabels) = self.getTicks(ticknum, lowerlimit, upperlimit)

        # plots a line according to (x,y) coordinates
        self.subplot[axName].plot(xvar, yvar, color=color, linewidth=width)
        self.subplot[axName].yaxis.set_major_locator(MultipleLocator(ticks))
        self.subplot[axName].yaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        self.subplot[axName].set_yticks(ticks)
        self.subplot[axName].set_yticklabels(tickLabels)
        self.subplot[axName].set_ylim(lowerlimit, upperlimit)

        if flip==True:
            self.subplot[axName].set_ylim(upperlimit*1.01, lowerlimit*1.01)
        else:
            self.subplot[axName].set_ylim(lowerlimit*1.01, upperlimit*1.01)
    
        for label in self.subplot[axName].get_xticklabels():
            label.set_visible(False)

    def setAxisLabels(self, axName, xmin, xmax, display=True):
        """
        aligns the x axis among all subplots
        """
        xlim = xmin, xmax
        self.subplot[axName].set_xlim(xlim)
        self.subplot[axName].xaxis.set_major_locator(self.majorxticks)
        self.subplot[axName].xaxis.set_minor_locator(self.minorxticks)
        self.subplot[axName].xaxis.set_major_formatter(DateFormatter(self.dateFmt))
        for label in self.subplot[axName].get_xticklabels():
                label.set_rotation(90)
                label.set_horizontalalignment('center')
                label.set_fontsize(8)
                label.set_visible(display)

    def reduce_opacity(self, im, opacity):
        """Returns an image with reduced opacity."""
        assert opacity >= 0 and opacity <= 1
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        else:
            im = im.copy()
        alpha = im.split()[3]
        alpha = ImageEnhance.Brightness(alpha).enhance(opacity)
        im.putalpha(alpha)
        return im

    def watermark(self, im, mark, position, opacity=1):
        """Adds a watermark to an image."""
        if opacity < 1:
            mark = self.reduce_opacity(mark, opacity)
        if im.mode != 'RGBA':
            im = im.convert('RGBA')
        # create a transparent layer the size of the image and draw the
        # watermark in that layer.
        layer = Image.new('RGBA', im.size, (0,0,0,0))
        if position == 'tile':
            for y in range(0, im.size[1], mark.size[1]):
                for x in range(0, im.size[0], mark.size[0]):
                    layer.paste(mark, (x, y))
        elif position == 'scale':
            # scale, but preserve the aspect ratio
            ratio = min(float(im.size[0]) / mark.size[0], float(im.size[1]) / mark.size[1])
            w = int(mark.size[0] * ratio)
            h = int(mark.size[1] * ratio)
            mark = mark.resize((w, h))
            layer.paste(mark, ((im.size[0] - w) / 2, (im.size[1] - h) / 2))
        else:
            layer.paste(mark, position)
        # composite the watermark with the layer
        return Image.composite(layer, im, layer)
