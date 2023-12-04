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


def addYear(df, dateCol):
    data = df
    data['year'] = data[dateCol].apply(lambda x: x.year)
    return data


def buildGains(gainType, gainPeriods):
    gains = []
    for p in gainPeriods:
        gains.append('%s%s' % (gainType, p))
    return gains


def filterReturnStats(gains):
    displayCols = []
    for ret in gains:
        displayCols.append('%s_count' % ret)
        displayCols.append('%s_win_ct' % ret)
        displayCols.append('%s_win_ratio' % ret)
        displayCols.append('%s_mean' % ret)
        displayCols.append('%s_ext_win_ct' % ret)
        displayCols.append('%s_ext_win_ratio' % ret)
        displayCols.append('%s_ext_loss_ct' % ret)
        displayCols.append('%s_ext_loss_ratio' % ret)
    return displayCols


def filterSignals(df, filterCols, dateCol, window, begin=None, end=None):
    data = df
    data['%s_epoch' % dateCol] = data[dateCol].map(
        lambda x: time.mktime(x.timetuple()))
    data = data.sort_values(filterCols + ['%s_epoch' % dateCol])
    data.reset_index(inplace=True)
    data['unique'] = data[filterCols].astype(
        str).apply(lambda x: '-'.join(x), axis=1)
    last = None
    signals = [False] * data.shape[0]
    for item, row in data.iterrows():
        curr = row['unique']
        if curr != 'nan':
            if last != curr:
                signals[item] = True
                lastDate = row[dateCol]
            else:
                nextDate = lastDate + datetime.timedelta(days=window)
                if row[dateCol] > nextDate:
                    signals[item] = True
                    lastDate = row[dateCol]
                    nextDate = lastDate + datetime.timedelta(days=window)
        last = curr
    if sum(signals) > 0:
        fdata = data[signals]
        fdata.drop(columns=['unique'])
        if begin is not None:
            fdata = fdata[fdata[dateCol] >= begin]
        if end is not None:
            fdata = fdata[fdata[dateCol] <= end]
    else:
        fdata = pd.DataFrame({})
    return fdata


# needs converted to pandas dataframe
def combine(var1, var2, level, weight=0.5):
    # combine 2 variables into 1 using a threshhold level
    # and a weighting to be applied to each variable
    maxZscore = level

    zsTuples = list(zip(var1, var2))

    combined = []
    combAppend = combined.append

    for zs in zsTuples:
        if min(zs) > 0:
            if min(zs) < maxZscore:
                if max(zs) >= maxZscore:
                    combAppend(min(zs) * weight + maxZscore * (1 - weight))
                else:
                    combAppend(zs[0] * weight + zs[1] * (1 - weight))
            else:
                combAppend(zs[0] * weight + zs[1] * (1 - weight))
        else:
            if max(zs) > maxZscore * -1:
                if min(zs) <= maxZscore * -1:
                    combAppend(max(zs) * weight + (
                        maxZscore * -1) * (1 - weight))
                else:
                    combAppend(zs[0] * weight + zs[1] * (
                        1 - weight))
            else:
                combAppend(zs[0] * weight + zs[1] * (
                    1 - weight))

    return combined


# needs converted to pandas dataframe
def seasonalize(col, data, period=63):
    outputColumns = []
    computed = {}

    # calc a 3 month mov avg, a centered avg of the mov avg,
    # a ratio of mov avg / centered, and an adjusted shares based on the ratio
    calcs = ('qtr_sma', 'ctr_sma', 'adjusted', 'adjratio')

    for var in calcs:
        computed[var] = {}

    s = RollingStats.RollingStats(data.value_dict[col], period)
    computed['qtr_sma'][col] = [x for x in s.sma]
    while len(computed['qtr_sma'][col]) < data.num_rows:
        computed['qtr_sma'][col].insert(0, 0)

    s = RollingStats.RollingStats(
        computed['qtr_sma'][col][period - 1:], period)
    computed['ctr_sma'][col] = [x for x in s.sma]

    while (data.num_rows - len(computed['ctr_sma'][col])) > period/2:
        computed['ctr_sma'][col].insert(0, 0)

    while len(computed['ctr_sma'][col]) < data.num_rows:
        computed['ctr_sma'][col].append(0)

    # the adjusted shares here is actually calced
    # by the actual values - the centered moving avg
    # it will br overwritten for values > 20050101
    # later as part of the actual estimation algo using an
    # avg of the ratio for a given week #
    computed['adjusted'][col] = np.array(
        data.value_dict[col]) - np.array(computed['ctr_sma'][col])
    computed['adjratio'][col] = np.array(
        computed['qtr_sma'][col]) / np.array(computed['ctr_sma'][col])

    colnames, cols = [], []

    for var in calcs:
        cols.append(computed[var][col])
        colnames.append(('%s_%s' % (col, var)))

    for i in range(len(colnames)):
        colname = colnames[i]
        outputColumns.append(colname)
        data.insert_column(colname, cols[i])

    # slice up the dataset in order to get the avg ratio for
    # the current week over the prior 3 years to use an an
    # estimate of the current ratio, make sure not
    # to include the current week
    # recalc out an estimate of adjusted share using
    # actual value - ( 3 month mov avg / ratio estimate )
    ratios = [0] * data.num_rows
    for i in range(data.num_rows):
        if data.value_dict['trade_date'][i] > 20050101:
            if data.value_dict['week'][i] < 52:
                weekSlice = data.where_field_equal(
                    'week', data.value_dict['week'][i])
            else:
                weekSlice = data.where_field_greaterequal('week', 52)

            date = datetime.datetime.strptime(
                str(data.value_dict['trade_date'][i]), '%Y%m%d')
            # bDate = int((date - datetime.timedelta(
            #    days=1460)).strftime('%Y%m%d'))
            eDate = int((date - datetime.timedelta(
                days=7)).strftime('%Y%m%d'))
            weekSlice = weekSlice.where_field_greater('trade_date', 20031110)
            weekSlice = weekSlice.where_field_less('trade_date', eDate)
            if weekSlice is not None:
                weekSlice = weekSlice.where_field_not_equal(
                    '%s_adjratio' % col, scipy.inf)
                if weekSlice is not None:
                    avgRatio = weekSlice.mean('%s_adjratio' % col)
                    if np.isnan(avgRatio):
                        avgRatio = 1
                    data.value_dict[
                        '%s_adjusted' % col][i] = data.value_dict[col][i] - (
                            data.value_dict['%s_qtr_sma' % col][i] / avgRatio)
                else:
                    avgRatio = 1
            else:
                avgRatio = 1
            ratios[i] = avgRatio
    data.insert_column('%s_estratio' % col, ratios)
    return outputColumns


def getGitPath():

    home = os.environ['DATA_ROOT']
    gitRevParse = subprocess.Popen(
        'git rev-parse --show-prefix',
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE)
    IsGitPrefix = gitRevParse.communicate()[0]
    gitPrefix = IsGitPrefix.rstrip()
    cwd = "/".join(gitPrefix.split('/')[1:])
    path = "%s/%s" % (home, cwd)

    return path


def createProject(directory, follow=False):
    home = os.environ['DATA_ROOT']
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


def appendPath(path):
    sys.path.reverse()
    sys.path.append(path)
    sys.path.reverse()


def product(lists):
    if not lists:
        yield []
    else:
        for item in lists[0]:
            for prod in product(lists[1:]):
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
    return (df[col].rank() / float(
        len(df[col][pd.notnull(df[col])]))*100).map(math.floor)


def percentileBins(df, column, bins):
    data = df
    divisor = 100/bins
    data['%s_rank' % column] = percentileRank(df, column)
    mask = data['%s_rank' % column] == 100
    data['%s_rank' % column][mask] = 99
    data['%s_bins' % column] = (
        data['%s_rank' % column]/divisor).map(lambda x: math.floor(x+1))
    return data


def groupValues(df, column, values):
    data = df
    data['%s_%s' % (column, 'group')] = None
    while(len(list(values.keys())) > 0):
        txt = list(values.keys())[0]
        minx = values[txt][0]
        maxx = values[txt][1]
        if minx is not None and maxx is None:
            mn = data[column] > minx
            mask = mn
        elif minx is None and maxx is not None:
            mx = data[column] <= maxx
            mask = mx
        else:
            mn = data[column] > minx
            mx = data[column] <= maxx
            mask = mn & mx
        data['%s_%s' % (column, 'group')][mask] = txt
        del values[txt]
    return data


def addMcapGroups(df, column, bins={}):
    if '%s_group' % column not in df.columns:
        mcaps = bins
        if bins == {}:
            mcaps = {
                'micro': (0, 2e8),
                "small": (2e8, 2e9),
                "mid": (2e9, 10e9),
                "10B+": (10e9, 50e9),
                "mega": (50e9, None)}
        df = groupValues(df, column, mcaps)
        df[column].fillna('no data')
    return df


def addPreGroups(df, prefix, period):

    column = '%s%s' % (prefix, period)
    if '%s_group' % column not in df.columns:
        if period == '6m':
            periods = {
                "-100->-37": (None, -37),
                "-37->-20": (-37, -20),
                "-20->-9": (-20, -9),
                "-9->0": (-9, 0),
                "0->14": (0, 14),
                "14->29": (14, 29),
                "29->50": (29, 50),
                "50+": (50, None)}
        else:
            # 3m default
            periods = {
                "-100->-26": (None, -26),
                "-26->-14": (-26, -14),
                "-14->-7": (-14, -7),
                "-7->0": (-7, 0),
                "0->10": (0, 10),
                "10->18": (10, 18),
                "18->32": (18, 32),
                "32+": (32, None)}
        df = groupValues(df, column, periods)
        df[column].fillna('no data')
    return df


def combineGroups(data, columns, length=2):
    combos = []
    for i in itertools.combinations(columns, length):
        combos.append(i)

    for combo in combos:
        newColName = " : ".join(combo)
        tmpData = '(' + data[combo[0]] + ')'
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

def detectOutliers(df, columns, zs=3.5, positive=None, lmt=None):
    df = df.copy()
    for col in columns:
        df['%s_outlier' % col] = 0
        med = df[col].median()
        madList = np.abs(df[col] - med)
        mad = madList.median()
        medianZS = 0.6745 * (df[col] - med)/mad
        if positive is None:
            pos = medianZS < zs
            neg = medianZS >= -zs
            mask = pos & neg
            mask = -mask
        else:
            if positive:
                multiplier = 1
                asc = False
            else:
                multiplier = -1
                asc = True

            data = medianZS.sort_values(ascending=asc)
            nrows = len(data)
            mask = [False] * nrows
            if lmt is None:
                lmt = nrows
            for i in range(lmt):
                if data.iloc[i] * multiplier > zs:
                    mask[i] = True

        df.loc[mask, '%s_outlier' % col] = 1

    return df

def augDickeyFuller(series, lag):
    stats = ts.adfuller(series, lag)
    return stats

def eventStats(
    data, event, columns, returns, outliers=False, summary=False,
    extremes={
        'post3m': 8.0,
        'post6m': 15.0,
        'post12m': 25.0,
        'rel3m': 5.0,
        'rel6m': 10.0,
        'rel12m': 15}
):
    if data.empty:
        raise RuntimeError('Cant run eventStats on an empty DataFrame.')

    columnOrder = ['event'] + columns
    sCols = [
        'count',
        'win_ct',
        'win_ratio',
        'median',
        'mean',
        'std',
        'min',
        'max',
        'ext_win_ct',
        'ext_win_ratio',
        'ext_loss_ct',
        'ext_loss_ratio']

    for ret in returns:
        for col in sCols:
            colname = '%s_%s' % (ret, col)
            columnOrder.append(colname)

    if type(event) == int:
        data = data.rename(columns={'evcode': 'event'})
    else:
         data = data.rename(columns={event: 'event'})

    for ret in returns:
        if not outliers:
            oCol = '%s_outlier' % ret
            if oCol in data.columns:
                data[ret][data[oCol] == 1] = np.nan
                df = data
            else:
                df = data
        else:
            df = data

        if summary:
            for col in columns:
                df[col] = col

    for ret in returns:
        df['%s_win_ct' % ret] = df[ret]
        df['%s_ext_win_ct' % ret] = df[ret] - extremes[ret]
        df['%s_ext_loss_ct' % ret] = df[ret] - (extremes[ret] * -1)
        win = df['%s_win_ct' % ret]
        pos = df['%s_ext_win_ct' % ret]
        neg = df['%s_ext_loss_ct' % ret]
        win[win <= 0] = np.nan
        pos[pos <= extremes[ret]] = np.nan
        neg[neg >= (extremes[ret] * -1)] = np.nan

    sdf = df[['event'] + columns + returns]
    grpsdf = sdf.groupby(['event'] + columns, as_index=False)
    sdf = grpsdf[returns].agg(
        [
            'count',
            'mean',
            'median',
            'std',
            'min',
            'max'
        ]).round(3).stack().reset_index().rename(
            columns={'level_2': 'stats',  'level_3': 'stats'})
    sdf = pd.pivot_table(
        sdf, index=['event'] + columns, values=returns, columns=['stats'])
    sdf.columns = [
        ' '.join(col).strip().replace(' ', '_') for col in sdf.columns.values]

    grpCols = []
    for ret in returns:
        grpCols.append('%s_win_ct' % ret)
        grpCols.append('%s_ext_win_ct' % ret)
        grpCols.append('%s_ext_loss_ct' % ret)

    sdf2 = df[['event'] + columns + grpCols]
    grpsdf2 = sdf2.groupby(['event'] + columns, as_index=False)
    sdf2 = grpsdf2[grpCols].agg(
        ['count']).round(3).stack().reset_index().rename(
            columns={'level_2': 'stats', 'level_3': 'stats'})
    sdf2 = pd.pivot_table(sdf2, index=['event'] + columns, columns=['stats'])
    sdf2.columns = sdf2.columns.get_level_values(0)
    sdf = sdf.merge(sdf2, how='inner', on=['event'] + columns)
    for ret in returns:
        sdf['%s_win_ratio' % ret] = \
            sdf['%s_win_ct' % ret] / sdf['%s_count' % ret]
        sdf['%s_ext_win_ratio' % ret] = \
            sdf['%s_ext_win_ct' % ret] / sdf['%s_count' % ret]
        sdf['%s_ext_loss_ratio' % ret] = \
            sdf['%s_ext_loss_ct' % ret] / sdf['%s_count' % ret]
    sdf.reset_index(inplace=True)
    return sdf[columnOrder]


def varianceCovariance(p, c, mu, sigma):
    alpha = norm.ppf(1-c, mu, sigma)
    return p - p*(alpha + 1)


def valueAtRisk(pv, ci, series):
    mu = np.mean(series)
    sigma = np.std(series)
    valueAtRisk = varianceCovariance(p, ci, mu, sigma)
    return valueAtRisk


def cusum(x, period, C=0.5, T=5, drift=0.25, target=None):
    stats = rollingstats.RollingStats(x, period)
    mean = stats.mean[period-1]
    if target is None:
        tgt = mean
    else:
        tgt = max(mean, target)
    std = stats.stddev[period-1]
    c = C * max(std, drift)
    t = T * max(std, drift)
    cUp = [0]
    cDown = [0]
    upSignals = []
    downSignals = []
    for i in range(period - 1, len(x)):
        cusumUp = max(0, cUp[-1] + stats.mean[i] - tgt - c)
        if cusumUp > t:
            upSignals.append(i)
        cUp.append(cusumUp)
        cusumDown = max(0, cDown[-1] + tgt - stats.mean[i] - c)
        if cusumDown < -t:
            downSignals .append(i)
        cDown.append(cusumDown)

    res = {}
    res['C'] = c
    res['T'] = t
    res['mean'] = mean
    res['target'] = target
    res['stddev'] = std
    res['drift'] = drift
    res['cusum_up'] = cUp
    res['cusum_down'] = cDown
    res['up_signals'] = upSignals
    res['down_signals'] = downSignals

    return res


def eventStatsExport(
    data, excel_file, sheet, event, gains, groups,
    col=0, row=0, flat_index=False
):
    display_cols = filterReturnStats(gains)
    stats = eventStats(data, event, groups, gains, summary=False)
    gstats = eventStats(data, event, groups, gains, summary=True)
    sdata = stats.append(gstats)
    sdata = sdata.reset_index()
    dropCols = ['level_0']
    for drop in dropCols:
        if drop in sdata.columns:
            sdata = sdata.drop(columns=[drop])
    sdata = sdata.groupby(['event'] + groups).sum()
    sdata.columns = [''.join(col).strip() for col in sdata.columns.values]
    if flat_index:
        indexCols = sdata.index.names
        flatIndex = np.array(sdata.index.to_flat_index().to_list())
        for i in range(0, len(indexCols)):
            sdata[indexCols[i]] = flatIndex[:, i]
        flatIndex = flatIndex.tolist()
        sdata = sdata.set_index(indexCols).reset_index()
    else:
        indexCols = []
    if len(sdata) > 0:
        excelCols = indexCols + display_cols
        sdata[excelCols].to_excel(
            excel_file,
            sheet_name=sheet,
            header=True,
            startcol=0,
            startrow=row)
        row += len(sdata.index) + 2
    excel_file.save()
    return row


def combineExcelFiles(globFiles, outputFile, sheets):     
    outputFile = pd.ExcelWriter(outputFile)
    excelFiles = glob.glob(globFiles)
    for sheet in sheets:
        row = 0
        for fn in excelFiles:
            data = pd.read_excel(fn, sheet_name=sheet)
            data.reset_index(inplace=True)
            for colname in ['level_0', 'index']:
                if colname in data.columns:
                    data = data.drop(columns=[colname])
            data.to_excel(
                outputFile,
                sheet_name=sheet,
                header=True,
                startcol=0,
                startrow=row)
            row += len(data.index) + 2
    outputFile.save()
