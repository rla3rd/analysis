import pandas as pd
import datetime
import numpy as np
import scipy as sp
import scipy.stats
import math
import itertools
import time
import os
import sys
import subprocess
import RollingStats
import statsmodels.tsa.stattools as ts

__version__ = '1.10.5'

class AttributeDict(dict):
    def __getattr__(self, attr):
        return self[attr]
    def __setattr__(self, attr, value):
        self[attr] = value

def calcIdxDays(frame):
    # get the idxdays distance between txns
    # used for tradedays signal
    last = frame['idxdate'].shift(1)
    frame['idxdays'] = frame['idxdate'] - last
    return frame

def filterSignals(df, filterCol, dateCol, window, begin=None, end=None):
    data = df
    data['%s_epoch' % dateCol] = data[dateCol].map(lambda x: time.mktime(x.timetuple()))
    data = data.sort(columns=[filterCol, '%s_epoch' % dateCol])
    results = {}
    #filterCol = 'ticker'
    #dateCol = 'eventdate'
    filters = data[filterCol].unique()
    for filter in filters:
        if filter != 'nan':
            fData = data[data[filterCol]==filter]
            signals = [True]
            if len(fData) > 1:
                idx = range(0, len(fData.index))
                lastDate = fData[dateCol].iloc[idx[0]]
                nextDate = lastDate + datetime.timedelta(days=window)
                for row in idx[1:]:
                    if fData[dateCol].iloc[row] > nextDate:
                        signals.append(True)
                        lastDate = fData[dateCol].iloc[row]
                        nextDate = lastDate + datetime.timedelta(days=window)
                    else:
                        signals.append(False)
            if len(fData[dateCol]) > 0:
                results[filter] = fData[signals]
    data = pd.concat(results, ignore_index=True)
    if begin != None:
        data = data[data[dateCol] >=begin]
    if end != None:
        data = data[data[dateCol] <= end]
    return data

def stats(var, tmpFilters, tmpReturns):

    results = {'total': None, 'win_ct': None, 'lose_ct': None, 'win_ratio': None, 
                    'lose_ratio': None, 'return_med': None, 'return_avg': None, 'return_stddev': None, 
                    'return_min': None, 'return_max': None, 'slope': None, 'intercept': None, 'r': None, 'r_low': None, 
                    'r_high': None, '2_tail_prob': None, 'std_err': None}
    
    total = float( len(tmpReturns) )
    winct = float( sp.greater( tmpReturns, 0 ).sum() )
    losect = total - winct
    if total > 0:
        winrt = winct / total
        losert = losect / total
    else:
        winrt = 0
        losert = 0
    if len(tmpReturns) > 0:
        returnMed = sp.median( tmpReturns )
        returnAvg = sp.mean( tmpReturns )
        returnStdDev = sp.std( tmpReturns )
        returnMin = np.min( tmpReturns )
        returnMax = np.max( tmpReturns )
    else:
        returnMed = 0
        returnAvg = 0
        returnStdDev = 0
        returnMin = 0
        returnMax = 0
    if total > 0 and var != None:
        r = scipy.stats.linregress( tmpFilters, tmpReturns )
        corr = r[2]
        z_r = np.arctanh(corr)
        ci = 1.96
        z_low = z_r - ci/np.sqrt(len(tmpReturns)-3)
        z_high = z_r + ci/np.sqrt(len(tmpReturns)-3)
        r_low = ( np.exp(1) ** ( 2 * z_low ) - 1 ) / ( np.exp(1) ** ( 2 * z_low ) + 1 )
        r_high = ( np.exp(1) ** ( 2 * z_high ) - 1 ) / ( np.exp(1) ** ( 2 * z_high ) + 1 )

        slope = r[0]
        intercept = r[1]
        twoTail = r[3]
        stdErr = r[4]

    else:
        corr = 0
        r_low = 0
        r_high = 0
        slope = 0
        intercept = 0
        twoTail = 0
        stdErr = 0
                    
    if len(tmpReturns) > 0:
        results =  { 'total': total, 
                    'win_ct': winct, 
                    'lose_ct': losect, 
                    'win_ratio': winrt, 
                    'lose_ratio': losert, 
                    'return_med': returnMed,
                    'return_avg': returnAvg, 
                    'return_stddev': returnStdDev, 
                    'return_min': returnMin,
                    'return_max': returnMax,
                    'slope': slope, 
                    'intercept': intercept, 
                    'r': corr,
                    'r_low': r_low, 
                    'r_high': r_high, 
                    '2_tail_prob': twoTail, 
                    'std_err': stdErr}
            
    return results

def summaryStats(df, filterColumns, returnColumns, regress=None, debug=False):
    if regress == None:
        regressionColumns = [None]
    else:
        regressionColumns = [None] + regress
    groupby={}
    data = df
    key = 'ALL'
    val = 'ALL'

    statColumns = ['return',
                    'total', 'win_ct', 'lose_ct', 'win_ratio', 'lose_ratio',
                    'return_med', 'return_avg', 'return_stddev', 'return_min', 'return_max']

    if regress != None:
        statColumns.reverse()
        statColumns.append('variable')
        statColumns.reverse()
        statColumns.append('slope')
        statColumns.append('intercept')
        statColumns.append('r')
        statColumns.append('r_low')
        statColumns.append('r_high')
        statColumns.append('2_tail_prob')
        statColumns.append('std_err')

    statColumns.reverse()
    statColumns.append('columnValue')
    statColumns.append('columnKey')
    statColumns.reverse()

    rows = []
    
    for returns in returnColumns:
        returnsData = data.dropna(subset=[returns])
        for var in regressionColumns:
            if var != None and len(returnsData)>0:
                if debug:
                    print key, val, var, returns
                regressionData = returnsData.dropna(subset=[var])
                tmpFilters = regressionData[var]
            else:
                regressionData = returnsData
                tmpFilters = []
                    
            tmpReturns = regressionData[returns]
            results = stats(var, tmpFilters, tmpReturns)
                   
            row = {'columnKey': key, 'columnValue': val, 'variable': var, 'return': returns}
            row.update(results)
                      
            rows.append(row)

    for col in filterColumns:
        g = data[col].unique()
        groupby[col] = filter( None, [v if pd.notnull(v) else None for v in g]) 
    
    keys = groupby.keys() 
    
    for key in keys:
        for val in groupby[key]:
            filteredData = data[data[key] == val]
            for returns in returnColumns:
                returnsData = filteredData.dropna(subset=[var])
                for var in regressionColumns:
                    if var != None:
                        regressionData = returnsData.dropna(subset=[var])
                        tmpFilters = regressionData[var]
                    else:
                        regressionData = returnsData
                        tmpFilters = []
    
                    tmpReturns = regressionData[returns]
                    results = stats(var, tmpFilters, tmpReturns)
    
                    row = {'columnKey': key, 'columnValue': val, 'variable': var, 'return': returns}
                    row.update(results)
    
                    rows.append(row)

    data = pd.DataFrame(rows, columns=statColumns)
    try:
        data['r'] = data['r'].real
        data['r_low'] = data['r_low'].real
        data['r_high'] = data['r_high'].real
    except:
        pass
    return data

#needs converted to pandas dataframe
def seasonalize( col, data, period=63 ):
    outputColumns = []
    computed = {}

    # calc a 3 month mov avg, a centered avg of the mov avg,
    # a ratio of mov avg / centered, and an adjusted shares based on the ratio
    calcs = ('qtr_sma', 'ctr_sma', 'adjusted', 'adjratio')

    for var in calcs:
        computed[var] = {}

    s = RollingStats.RollingStats( data.value_dict[col], period)
    computed['qtr_sma'][col] = [ x for x in s.sma ]
    while len(computed['qtr_sma'][col]) < data.num_rows:
        computed['qtr_sma'][col].insert(0,0)

    s = RollingStats.RollingStats( computed['qtr_sma'][col][period - 1:], period)
    computed['ctr_sma'][col] = [ x for x in s.sma ]

    while ( data.num_rows - len(computed['ctr_sma'][col]) ) > period/2:
        computed['ctr_sma'][col].insert(0,0)

    while len(computed['ctr_sma'][col]) < data.num_rows:
        computed['ctr_sma'][col].append(0)

    # the adjusted shares here is actually calced by the actual values - the centered moving avg
    # it will br overwritten for values > 20050101 later as part of the actual estimation algo using an
    # avg of the ratio for a given week #
    computed['adjusted'][col] = scipy.array( data.value_dict[col] ) - scipy.array( computed['ctr_sma'][col] )
    computed['adjratio'][col] = scipy.array( computed['qtr_sma'][col] ) / scipy.array( computed['ctr_sma'][col] )

    colnames, cols = [], []

    for var in calcs:
        cols.append(computed[var][col])
        colnames.append(('%s_%s' % (col, var)))

    for i in range(len(colnames)):
        colname = colnames[i]
        outputColumns.append(colname)
        data.insert_column(colname, cols[i])

    # slice up the dataset in order to get the avg ratio for the current week over the prior 3 years to use an an
    # estimate of the current ratio, make sure not to include the current week
    # recalc out an estimate of adjusted share using actual value - ( 3 month mov avg / ratio estimate )
    ratios = [0] * data.num_rows
    for i in range(data.num_rows):
        if data.value_dict['trade_date'][i] > 20050101:
            if data.value_dict['week'][i] < 52:
                weekSlice = data.where_field_equal('week', data.value_dict['week'][i])
            else:
                weekSlice = data.where_field_greaterequal('week', 52)

            date = datetime.datetime.strptime( str(data.value_dict['trade_date'][i]), '%Y%m%d' )
            bDate = int((date - datetime.timedelta(days=1460)).strftime('%Y%m%d'))
            eDate = int((date - datetime.timedelta(days=7)).strftime('%Y%m%d'))
            weekSlice = weekSlice.where_field_greater('trade_date', 20031110)
            weekSlice = weekSlice.where_field_less('trade_date', eDate )
            if weekSlice != None:
                weekSlice = weekSlice.where_field_not_equal('%s_adjratio' % col, scipy.inf)
                if weekSlice != None:
                    avgRatio = weekSlice.mean('%s_adjratio' % col)
                    if scipy.isnan( avgRatio ):
                       avgRatio = 1
                    data.value_dict['%s_adjusted' % col][i] = data.value_dict[col][i] - ( data.value_dict['%s_qtr_sma' % col][i] / avgRatio )
                else:
                    avgRatio = 1
            else:
                avgRatio = 1
            ratios[i] = avgRatio
    data.insert_column('%s_estratio' % col, ratios )
    return outputColumns

def getGitPath():

    home = os.environ['DR']
    gitRevParse = subprocess.Popen('git rev-parse --show-prefix', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    IsGitPrefix = gitRevParse.communicate()[0]
    gitPrefix = IsGitPrefix.rstrip()
    cwd = "/".join( gitPrefix.split('/')[1:] )
    path = "%s/%s" % ( home, cwd )

    return path

def createProject(directory, follow=False):
    home = os.environ['DR']
    cwd = os.getcwd()
    os.chdir('%s/analysis/projects' % home)
    os.mkdir(directory)
    os.chdir(directory)
    os.mkdir('notebooks')
    os.mkdir('data')
    os.mkdir('scripts')
    os.mkdir('results')
    if not follow:
        os.chdir(cwd)
    else:
        os.chdir('scripts')

def appendPath( path ):
    sys.path.reverse()
    sys.path.append( path )
    sys.path.reverse()

def product( lists ):
    if not lists:
        yield []
    else:
        for item in lists[0]:
            for prod in product( lists[1:]):
                yield [item] + prod

def ifelse(condition, a, b):
    """
    This is a small if/else function for use in lambda functions.
    """

    if condition:
        return a
    else:
        return b

def percentileRank(df, col):
    """
    This function calcs the percentile rank row in the column
    """
    return (df[col].rank() / float(len(df[col][pd.notnull(df[col])==True]))*100).map(math.floor)

def percentileBins(df, column, bins):
    data = df
    divisor = 100/bins
    data['%s_rank' % column]  = percentileRank(df, column)
    mask = data['%s_rank' % column] == 100
    data['%s_rank' % column][mask] = 99
    data['%s_bins' % column] = (data['%s_rank' % column]/divisor).map(lambda x: math.floor(x+1))
    return data

def groupValues(df, column, values):
    data = df.copy()
    data['%s_%s' % (column, 'group')] = None
    values = values.copy()
    while(len(values.keys()) > 0):
        txt = values.keys()[0]
        minx = values[txt][0]
        maxx = values[txt][1]
        if minx != None and maxx == None:
            mn = data[column] > minx
            mask = mn
        elif minx == None and maxx != None:
            mx = data[column] <= maxx
            mask = mx
        else:
            mn = data[column] > minx
            mx = data[column] <= maxx
            mask = mn & mx
        data['%s_%s' % (column, 'group')][mask] = txt
        del values[txt]
    return data

def combineGroups(data, columns, length=2):
    combos = []
    for i in itertools.combinations(columns, length):
        combos.append(i)

    for combo in combos:
        newColName = " : ".join(combo)
        tmpData = '(' + data[combo[0]] +')'
        cols = combo[1:]
        for col in cols:
            tmpData = tmpData + ':(' + data[col] + ')'
        data[newColName] = tmpData
    return data

def removeOutliers(df, column, lmt=None, nonzero=False, zs=3.5, positive=True):
    if positive:
        multiplier = 1
        asc = False
    else:
        multiplier = -1
        asc = True
    nulls = df[pd.notnull(df[column])==False]
    data = df[pd.notnull(df[column])==True]
    if nonzero == True:
        data = data[data[column] != 0]
    med = data[column].median()
    madList = abs(data[column] - med)
    mad = madList.median()
    if mad != 0:
        data['medianzs'] = 0.6745 * ( data[column] - med )  / mad
    else:
        data['medianzs'] = 0

    dfs = []
    if lmt == None:
        pos = data['medianzs'] <= zs
        neg = data['medianzs'] > -zs
        mask = pos & neg
        data = data[mask]
        dfs.append(data)
        dfs.append(nulls)
        outliers = data[-mask]
    else:
        data = data.sort(['medianzs'], ascending=asc)
        outList = []
        for i in range(lmt+1, 0, -1)[1:]:
            if data.head(1)['medianzs'] * multiplier > zs:
                outList.append(data.head(1))
                data = data.tail(len(data)-1)
            else:
                break

        dfs.append(data)
        dfs.append(nulls)
        data = pd.concat(dfs)
        if len(outList) > 0:
            outliers = pd.concat(outList)
        else:
            outliers = pd.concat([pd.DataFrame(outList)])
    data.pop('medianzs')
    return (data, outliers)

def groupbyStats(data, columns, returns, regress=None):
    columnOrder = ['return', 'total', 'win_ct', 'lose_ct', 'win_ratio', 'lose_ratio', 'return_med', 'return_avg', 'return_stddev', 'return_min', 'return_max']
    if regress!=None:
        columnOrder.reverse()
        columnOrder.append('variable')
        columnOrder.reverse()
        columnOrder.append('slope')
        columnOrder.append('intercept')
        columnOrder.append('r')
        columnOrder.append('r_low')
        columnOrder.append('r_high')
        columnOrder.append('2_tail_prob')
        columnOrder.append('std_err')
    frames = []
    grouped = data.groupby(columns)
    for name, group in grouped:
        if len(columns) >1:
            index = pd.MultiIndex.from_tuples([name], names=columns)
        else:
            index = pd.Index([name], name=columns[0])
        means = group.mean()
        medians = group.median()
        std = group.std()
        mins = group.min()
        maxs = group.max()
        for ret in returns:
            totals = ((group.count())[ret]).__float__()
            totalpos = ((group[group[ret]>0].count())[ret]).__float__()
            totalneg = ((group[group[ret]<=0].count())[ret]).__float__()
            tot = {'name': name, 'total': totals, 'win_ct': totalpos, 'lose_ct': totalneg}
            if tot['total'] > 0:
                tot['win_ratio'] = tot['win_ct']/tot['total']
                tot['lose_ratio'] = tot['lose_ct']/tot['total']
            else:
                tot['win_ratio'] = 0
                tot['lose_ratio'] = 0
            tot['return_med'] = medians[ret]
            tot['return_avg'] = means[ret]
            
            tot['return_stddev'] = std[ret]
            tot['return_min'] = mins[ret]
            tot['return_max'] = maxs[ret]
            tot['return'] = ret
            if regress != None:
                for var in regress:
                    tot['variable'] = var
                    corr = 0
                    r_low = 0
                    r_high = 0
                    slope = 0
                    intercept = 0
                    twoTail = 0
                    stdErr = 0
                    if len(group) > 1:
                        regGroup = group[group.apply(lambda x: pd.notnull(x[ret])==True, axis=1)]
                        if len(regGroup) >1:
                            regGroup = regGroup[regGroup.apply(lambda x: pd.notnull(x[var])==True, axis=1)]
                            if len(regGroup) > 1:
                                r = scipy.stats.linregress(regGroup[var], regGroup[ret])
                                #print name, var, ret, r
                                corr = r[2] 
                                if r[2] == 1:
                                    z_r = sp.inf
                                else:
                                    z_r = np.arctanh(corr)
                                ci = 1.96
                                z_low = z_r - ci/np.sqrt(len(regGroup)-3)
                                z_high = z_r + ci/np.sqrt(len(regGroup)-3) 
                                r_low = ( np.exp(1) ** ( 2 * z_low ) - 1 ) / ( np.exp(1) ** ( 2 * z_low ) + 1 )
                                r_high = ( np.exp(1) ** ( 2 * z_high ) - 1 ) / ( np.exp(1) ** ( 2 * z_high ) + 1 )
                        
                                slope = r[0]
                                intercept = r[1]
                                twoTail = r[3]
                                stdErr = r[4]
                                
                    tot['slope'] = slope
                    tot['intercept'] = intercept
                    tot['r'] = corr
                    tot['r_low'] = r_low
                    tot['r_high'] = r_high
                    tot['2_tail_prob'] = twoTail
                    tot['std_err'] = stdErr
                    frames.append(pd.DataFrame(tot, columns=columnOrder, index=index))
            else:
                frames.append(pd.DataFrame(tot, columns=columnOrder, index=index))
    
    df = pd.concat(frames)
    df.fillna(0)
    try:
        df['r'] = df['r'].real
        df['r_low'] = df['r_low'].real
        df['r_high'] = df['r_high'].real
    except:
        pass

    return df

def augDickeyFuller(series, lag):
    stats = ts.adfuller(series, lag)
    return stats

def hurst(series):
    lags = range(2,100)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]*2.0

def residuals(series1, series2):
    ols = pd.stats.api.ols
    results = ols(x=series1, y=series2)
    betaHedgeRatio = results.beta.x
    residuals = series2 - betaHedgeRatio * series1
    return residuals

def varianceCovariance(p, c, mu, sigma):
    alpha = norm.ppf(1-c, mu, sigma)
    return p - p*(alpha + 1)

def valueAtRisk(pv, ci, series):
    mu = np.mean(series)
    sigma = np.std(series)
    valueAtRisk = varianceCovariance(p, ci, mu, sigma)
    return valueAtRisk

class Stats:
    def  __init__(self, inList, period ):
        self.inList = ma.array(inList)
        self.stddev = []
        self.sma = []
        self.zscore = []
        smaAppend = self.sma.append
        stdAppend = self.stddev.append
        zsAppend = self.zscore.append
        xsum = 0
        x2sum = 0
        for i in xrange( period ):
            xsum += float( self.inList[i] )
            x2sum += float( self.inList[i] ** 2 )
            # print i, x2sum, xsum, period, self.inList[i-period], self.inList[i]
        mean = xsum / period
        smaAppend( mean )
        variance = round( ( x2sum - (xsum ** 2 )/ period ) / period, 4 )
        stddev = np.sqrt( variance )
        stdAppend( stddev )
        if stddev != 0:
            zs = ( float( self.inList[i] ) - mean ) / stddev
        else:
            zs = 0
        zsAppend( zs )
        for i in xrange( period , len(inList) ):
            xsum += float( self.inList[i] )- float( self.inList[i-period] )
            x2sum += float( self.inList[i] ) ** 2 - float( self.inList[i-period] ) ** 2
            mean = xsum / period
            smaAppend( mean )
            variance = round( ( x2sum - (xsum ** 2 )/ period ) / period, 4 )
            stddev = np.sqrt( variance )
            stdAppend( stddev )
            if stddev != 0:
                zs = ( float( self.inList[i] ) - mean ) / stddev
            else:
                zs = 0
            zsAppend( zs )
            # print i, x2sum, xsum, period, stdDevSq, self.inList[i-period], self.inList[i]

def wma( inList, period, nonzero = False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    avgList = []
    append = avgList.append
    for i in xrange( period-1, len(inList) ):
        slice = inList[startIdx:i+1]
        if nonzero == True:
            divisor = (len(slice.nonzero()[0])) * (len(slice.nonzero()[0])+1)/2
        else: 
            divisor = (len(slice) * (len(slice)+1) )/2
        numerator = 0
        mult = 1
        if nonzero == True:
            slice = [item for item in slice if item != 0]
        for item in slice:
            numerator += item * mult
            mult +=1
        append( np.float64(numerator) / np.float64(divisor) )
        startIdx +=1
    avgList = [ifelse(np.isnan(x) == False, x, 0) for x in avgList]
    return avgList

def wma_stddev( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    nz = nonzero
    stddevList = []
    append = stddevList.append
    for i in xrange( period-1, len(inList) ):
        slice = inList[startIdx:i+1]
        avg = wma(slice, len(slice), nonzero=nz)
       
        if nonzero == True:
            slice = [item for item in slice if item != 0]
        slice = np.array(slice)
        if len(slice) != 0:
            stddev = sum( ( slice - avg[-1] ) ** 2 ) / len(slice)
        else:
            stddev = 0
        append( stddev )
    return stddevList

def sma( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    avgList = []
    append = avgList.append
    for i in xrange( period-1, len(inList)):
        slice = inList[startIdx:i+1]
        if nonzero == True:
            divisor = len(slice.nonzero()[0])
        else:
            divisor = len(slice)
        numerator = sum(slice)
        append( np.float64(numerator) / np.float64(divisor) )
        startIdx +=1
    avgList = [ifelse(np.isnan(x) == False, x, 0) for x in avgList]
    return avgList

def stddev( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    avgList = []
    append = avgList.append
    for i in xrange( period-1, len(inList)):
        slice = inList[startIdx:i+1]
        if nonzero == True:
            slice = ma.masked_object(slice, 0)
        append( slice.std() )
        startIdx +=1
    return avgList

def sign( number ):
    return np.sign( number )

def min( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    minList = []
    append = minList.append
    for i in xrange ( period-1, len(inList)):
        slice = inList[startIdx:i+1]
        if nonzero == True:
            slice = ma.masked_object(slice, 0)
        append( slice.min() )
        startIdx += 1
    return minList

def max( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    maxList = []
    append = maxList.append
    for i in xrange ( period-1, len(inList)):
        slice = inList[startIdx:i+1]
        if nonzero == True:
            slice = ma.masked_object(slice, 0)
        append( slice.max() )
        startIdx += 1
    return maxList

def vol( inList, period, nonzero=False ):
    if nonzero == True:
        inList = ma.array(inList)
    else:
        inList = np.array(inList)
    startIdx = 0
    volList = []
    append = volList.append
    for i in xrange( period-1, len(inList) ):
        slice = inList[ startIdx : i+1 ]
        if nonzero == True:
            slice = ma.masked_object( slice, 0 )
        today = slice[1:]
        yesterday = slice[:-1]
        append( np.std( np.exp( np.log( today / yesterday ) ) ) * np.sqrt( 252 ) )
        startIdx +=1
    return volList

def validateCusip(cusip):
    if len(cusip) != 9:
        return False
    else:
        total= 0
        cusip = cusip.upper()
        alphabet="ABCDEFGHIJKLMNOPQRSTUVWXYZ*@#"
        for i in range(len(cusip)-1):
            if not cusip[i].isdigit():
                val =  alphabet.index(cusip[i])+1
            else:
                val = int(cusip[i])
            if i % 2 != 0 :
                val *= 2

            val = (val % 10) + (val / 10);
            total += val;
        check = (10 - (total % 10)) % 10
        if check == int(cusip[-1]):
            return True
        else:
            return False
