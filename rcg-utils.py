"""
Useful functions and classes
"""

import random
import datetime as dt
import numpy as np
import re
import math
import scipy.stats as sps



def listN(indict,N=10,rand=False):
    """
    Print first N items from a dictionary.  Can use 'rand=True' to look at a random selection of dictionary elements.
    """
    if rand:
        samp=random.sample(range(len(indict)),min([N,len(indict)]))
    else:
        samp=range(N)
    for i in samp:
        print str(list(indict)[i])+':'+str(indict[list(indict)[i]])


def selrand(x,N=10,replace=False,ordered=False):
    """
    Returns N instances of iterable x at random, returns at list.  Can choose whether to use replacement.  If not, N is bounded at len(x) to avoid ValueError.
    The 'ordered' keyword will turn on ordering, this is done by sorting indices, not output.
    """
    x=list(x)
    if replace:
        ix = np.random.randint(0,len(x),N)
    else:
        ix = random.sample(range(len(x)),min([N,len(x)]))
    outlst=[]
    if ordered: ix=sorted(ix)
    for i in ix:
        outlst.append(x[i])
    return outlst


def to_ones(inarr,center=0):
    """
    Takes array, returns array of -1, 0, and 1, based on whether each element is smaller, equal to, or larger than some specified center 
    so [-34,2,0,18,-5] becomes [-1,1,0,1,-1]
    """
    outarr=[]
    for item in inarr:
        if item>center: outarr.append(1)
        if item<center: outarr.append(-1)
        if item==center:outarr.append(0)
    return outarr


def list_count(inlst,declist=False):
    """
    returns census counts of distinct items in list as dict
    if declist set, will return as list of tuples in descending popularity
    """
    inlst=list(inlst)
    
    res = {item:inlst.count(item) for item in set(inlst)}
    
    if declist:
        return sorted([(x,res[x]) for x in res],key=lambda x: x[1],reverse=True)
    else:
        return res
        
def census(inlst,declist=False):
	return list_count(inlst,declist)




####Date and Time Functions#######
dmon={'JAN':[1,31,31],'FEB' : [2,28,31],'MAR' : [3,31,28],'APR' : [4,30,31],'MAY' : [5,31,30],'JUN' : [6,30,31],\
    'JUL' : [7,31,30],'AUG' : [8,31,31],'SEP' : [9,30,31], 'OCT' : [10,31,30],'NOV' : [11,30,31],'DEC' : [12,31,30]}

# convert slash format to standard date format string: todt(instr).strftime('%Y-%m-%d')

def todt(indate):
    if re.search('\d\d{2,}-\d{1,2}-\d{1,2} \d\d:\d\d:\d\d',indate):
        datestr,timestr=indate.split(' ')[0:2]
        dates=datestr.split('-')
        times=timestr.split(':')
        return dt.datetime(int(dates[0]),int(dates[1]),int(dates[2]),int(times[0]),int(times[1]),int(times[2].split('.')[0]))
    
    if re.search('\d{1,2}/\d{1,2}/\d{2,4}',indate):
        ds=indate.split('/')
        return dt.datetime(int(ds[2]),int(ds[0]),int(ds[1]),0,0,0)
    
    if re.search('\d{4}-\d{1,2}-\d{1,2}',indate):
        ds=indate.split('-')
        return dt.datetime(int(ds[0]),int(ds[1]),int(ds[2]),0,0,0)

    if re.search('\d\d[A-Za-z]{3}\d{2}',indate) and indate[2:5].upper() in dmon:
        return dt.datetime(int(indate[5:]),dmon[indate[2:5].upper()][0],int(indate[0:2]),0,0,0)
        

def dur(indate1,indate2):
    """
    Returns duration in sec
    """
    return (todt(indate1)-todt(indate2)).total_seconds()

def durday(indate1,indate2):
    """
    Returns duration in days
    """
    return int(math.floor((1.+(todt(indate2)-todt(indate1)).total_seconds())/86400.))
    
def date_fix(indate_str):
    """
    attempt to convert date string to standard YYYY-MM-DD format
    returns original string if that attempt fails
    """
    try:
        dt_obj = todt(indate_str)
        return dt_obj.strftime("%Y-%m-%d")
    except:
        #print 'WARNING, datetime conversion failed'
        return indate_str
    


### Percentile Functions ###
def p105090(lst):
    a,b,c=np.percentile(lst,[10,50,90])
    return str(a)[0:4]+'/'+str(b)[0:4]+'/'+str(c)[0:4]

def p255075(lst,abbr=False):
    a,b,c=np.percentile(lst,[10,50,90])
    if abbr: return str(3.0*b+c)[0:6]
    else: return str(a)[0:6]+'|'+str(b)[0:6]+'|'+str(c)[0:6]

def px3(lst):
    a,b,c=np.percentile(lst,[10,50,90])
    #return str(a)[0:4]+'/'+str(b)[0:4]+'/'+str(c)[0:4]+'/'+str(d)[0:4]+'/'+str(e)[0:4]
    return [a,b,c]

def px5(lst):
    a,b,c,d,e=np.percentile(lst,[10,25,50,75,90])
    f=np.std(lst)
    #return str(a)[0:4]+'/'+str(b)[0:4]+'/'+str(c)[0:4]+'/'+str(d)[0:4]+'/'+str(e)[0:4]
    return [a,b,c,d,e,f]

def px9(lst):
    a,b,c,d,e,f,g,h,i=np.percentile(lst,[10,20,30,40,50,60,70,80,90])
    return str(a)[0:4]+'/'+str(b)[0:4]+'/'+str(c)[0:4]+'/'+str(d)[0:4]+'/'+str(e)[0:4]+'/'+str(f)[0:4]+\
            '/'+str(g)[0:4]+'/'+str(h)[0:4]+'/'+str(i)[0:4]


###Plotting Help####
def tocu(lst,div=100,onestart=False):
    """
    Creates a two-part list which can be used for make a cumulative distro function from input list
    reads in list x, returns a twopart list y=[y1,y2]
    Can then plot output as plt.plot(y[0],y[1])
    """
    minx = min(lst)
    maxx = max(lst)
    lenx = len(lst)
    x = sorted(lst)
    ii=0
    outlst=[[minx],[int(onestart)]]
    for i in range(div):
        while x[ii]<minx+(float(i)/div)*(float(maxx)-minx):
            ii+=1
        outlst[0].append(minx+(float(i)/div)*(float(maxx)-minx))
        if onestart: outlst[1].append(1.0-float(ii)/lenx)
        else: outlst[1].append(float(ii)/lenx)
            
    return outlst
           
### Geography Functions####
def globe_dist(lat1, long1, lat2, long2):

    erad = 6371. #km
    degrees_to_radians = 3.14159265658979/180.0
        
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians
    
    cos = np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) + np.cos(phi1)*np.cos(phi2)
    arc = np.arccos( cos )

    return arc*erad


def flat_dist(lat1, long1, lat2, long2):  #Faster than globe dist, not as accurate for long distances
    erad = 6371. #km
    degrees_to_radians = 3.14159265658979/180.0

    llat = (lat1+lat2)/2.
    arc = ((lat1-lat2)**2 + ((long1-long2)*np.cos(llat*degrees_to_radians))**2)**0.5
    if np.isnan(arc):return -1
    return arc*degrees_to_radians*erad


### String Function
def dequoter(x):
    """
    returns string x without quotes or commas inside quotes.  Turns '"a, man", "a, plan"' into 'a man, a plan'
    """
    qcnt=0
    newx=''
    for item in x:
        if item==',':
            if qcnt%2==1:
                continue
            else:
                newx = newx + item
        if item=='"':
            qcnt+=1
        if item not in [',','"']:
             newx = newx + item

             
def fill_zeroes(x,digits):
    """
    read in int x, returns string of length digits, with leading 0's added if needed
    fill_zeroes(567,5) returns '00567'
    """
    dfill = ''.join(['0'])*digits
    return (dfill+str(x))[-digits:]


             
## Other Utilities
def counter(i,num):
    """
    prints i when it is divisible by num
    """
    if i%num==0: print i

def get_pwd(path):
    """ get password from file in path.  password should be on first line, no spaces"""
    with open(path,'r') as infile:
        line = infile.readline()
        return line.replace('\r','').replace('\n','').replace(' ','')
       


        
    
### Analysis

def ctest(n1, x1, n2, x2, c=1.):
    """
    implements a c-test for comparing means of 2 poisson distributions.  Based on eqn. 2.2 in 
    Krishnamoorthy and Thomson, Journal of Statistical Planning and Inference 119 (2004) 23 - 35
        n1, n2 = intervals
    x1, x2 = events in each interval
    c = desired ratio to test
     
    returns p-value for L1/L2 < c, where L1 and L2 are rates for x1 and x2.
    
    Be aware that this is based on the binomial distrobution, and has large errors at low values in x.
    
    I added the second return statement, which seems to improve accuracy at low numbers...?
    """
    a = (float(n1)/float(n2))*c
    pc = a/(1.+a)
    k = x1+x2
    
    #return sps.binom.cdf(k, k, pc, loc=0) - sps.binom.cdf(x1, k, pc, loc=0)
    return sps.binom.cdf(k, k, pc, loc=0) - 0.5*(sps.binom.cdf(x1, k, pc, loc=0)+sps.binom.cdf(x1-1, k, pc, loc=0))


### Pandas Utilities

def colconv_int(df,col):
    """
    returns a dataframe with column col converted to int, drops any rows that fail to convert
    """
    df[col] = df[col].convert_objects(convert_numeric=True)
    df = df[pd.notnull(df[col])]
    df[col] = df[col].apply(lambda x:int(x))
    return df


### Spark Utilities

def csv2parquet(inpath,outpath,hdfs_input=False):
    """
    converts a csv (local or hdfs, as specified by 3rd arg) to a parquet file.  Csv must have a header line at top, and should be column delimited.
    Need to add functionality to infer schema for hdfs load case.
    """
    if not outpath:
        outpath = inpath.replace('.csv','') + '.parquet'
        
    if outpath[-8:] != '.parquet': 
        outpath = outpath + '.parquet'
    
    if not hdfs_input:
        df = pd.read_csv(inpath)    
        sdf = sqlContext.createDataFrame(sc.parallelize(df.T.to_dict().values()))
        print sdf.take(3)
        sdf.write.parquet(outpath)
    
    else:
        df_rdd = sc.textFile(inpath).map(lambda line: line.split(","))
        header = df_rdd.first() #extract header
        df_rdd = df_rdd.filter(lambda row: row != header)   #filter out header
        sdf = sqlContext.createDataFrame(df_rdd, header)
        sdf.write.parquet(outpath)
        print sdf.take(3)


