#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
import empyrical as ep
from datetime import datetime, timedelta


# In[33]:


from datetime import datetime
from dateutil.relativedelta import relativedelta


# In[34]:


from __future__ import annotations
from pandas_market_calendars import get_calendar


# In[35]:


#设计一个观察日的函数
def get_same_day_dates(start_date, n ,lock):
    dates = []
    begin_date=datetime.strptime(start_date, '%Y-%m-%d')
    for i in range(lock,n+1):
        print(i)
        current_date = begin_date
        time=i
        current_date += relativedelta(months=time)  # 将日期增加一个月
        dates.append(current_date.strftime('%Y-%m-%d'))  # 将日期对象转换为字符串格式并添加到列表中
    return dates


# In[36]:


from pandas_market_calendars import get_calendar

# 获取上交所交易日历对象
sh_exchange_calendar = get_calendar('XSHG')

# 指定开始日期和结束日期
start_date = pd.to_datetime('2023-07-11')
end_date = pd.to_datetime('2026-12-31')

# 获取交易日日期列表
trading_days = sh_exchange_calendar.valid_days(start_date=start_date, end_date=end_date)

# 将交易日日期列表转换为字符串格式
trading_days_list = trading_days.strftime('%Y-%m-%d').tolist()

#将两个Dataframe拼凑在一起
datedf=pd.read_excel('中证1000指数_20230711_164100.xlsx')
datedf = datedf['指标名称']
datedf = pd.to_datetime(datedf)
datedf = datedf.dt.strftime("%Y-%m-%d")
new_list = datedf.tolist()
trading_days_list = new_list+trading_days_list


# In[37]:


pd.DataFrame(trading_days_list).to_excel('tradingdate.xlsx')


# In[38]:


#若为非交易日,则往后顺延到最近的交易日

def adjust_to_trading_days(dates, trading_days_list):
    adjusted_dates = []

    trading_days = pd.to_datetime(trading_days_list)

    for date in dates:
        dt = pd.to_datetime(date)

        if dt in trading_days:
            adjusted_dates.append(date)
        else:
            next_trading_day = trading_days[trading_days > dt].min()
            adjusted_dates.append(next_trading_day.strftime('%Y-%m-%d'))

    return adjusted_dates


# In[39]:


#形成最终的结果函数
#n代表产品的期限，lock代表锁定的月度数，先以12月期，锁3个月的产品进行分析,strike代表敲出价格是期初价格的百分比,r代表敲出后的收益率,begin_date
#代表开始日期，end_date代表终止日期,interrupt_date代表截断日期，roll_length代表PB,PE分位数向前滚动的天数
def snowball_dual(df_all,n,lock,strike,r,df_pbpe,begin_date,end_date,interrupt_date,roll_length):
    
    df_pbpe['PB分位数']=df_pbpe['PB'].rolling(roll_length).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df_pbpe['PE分位数']=df_pbpe['PE'].rolling(roll_length).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    df=df_all
    df.columns=['date','标的指数收盘价']
    df.index=df['date']
    df['敲出价格'] = strike*df['标的指数收盘价']
    
    #新增一列记录是否发生敲出
    df['是否敲出']=0
    
    #新增一列记录敲出或者到期或者截断的时间
    df['敲出或者到期时间']=0
    
    #新增一列记录产品敲出时指数点位
    df['敲出或到期时指数点位']=0
    
    #新增一列填入产品收益率
    df['收益率']=0
    
    #新增一列计算当前指数点位的历史分位数（滚动周期与PBPE计算周期相同）
    df['指数点位历史分位数']=df['标的指数收盘价'].rolling(roll_length).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    #新增一列判断产品是否存续
    df['产品当前是否存续']=0
    
    #截取需要部分
    df=df.loc[begin_date:end_date]
    df_pbpe=df_pbpe.loc[begin_date:end_date]

    #将PB/PE的数据读出并存入
    df_part1=pd.concat([df, df_pbpe], axis=1)
    print(df_part1)
    
    #选取日截至到interrupt_date
    df_return=df_part1.loc[:interrupt_date]
    print(df_return)
    
    
    #判断是否敲出并对以上信息进行填充
    for k in range(len(df_return)):
        start_date=df_return.iloc[k,0]
        dates=get_same_day_dates(start_date, n ,lock)
        obs_date_list = adjust_to_trading_days(dates,trading_days_list)
        obs_date=[date for date in obs_date_list if datetime.strptime(date, "%Y-%m-%d") <= datetime.strptime(end_date, "%Y-%m-%d")]
      
        if len(obs_date_list)>len(obs_date):
            df_return.iloc[k,8]= 1
          
        #提取观察日对应的标的指数点位
        price_obs_date= df_part1.loc[obs_date,'标的指数收盘价']
      
        #找到敲出的时间点
        knock_price=df_return.iloc[k,2]
        
        position = None
        for i, price in enumerate(price_obs_date):
            if price > knock_price:
                position = i
                df_return.iloc[k,3]=1
                #如果发生敲出，记录产品收益率
                df_return.iloc[k,6]=r
                break

        if position is None:
            position = len(price_obs_date) - 1   
    
        #记录敲出或者到期的时间
        df_return.iloc[k,4]=obs_date[position]
        print(obs_date[position])
        
        #记录敲出或到期时指数点位
        df_return.iloc[k,5]=df_part1.loc[obs_date[position],'标的指数收盘价']
    
    
    # 计算两个日期之间的差距
    df_return['date'] = pd.to_datetime(df_return['date'])
    df_return['敲出或者到期时间'] = pd.to_datetime(df_return['敲出或者到期时间'])
    df_return['产品存续天数'] = (df_return['敲出或者到期时间'] - df_return['date']).dt.days
             
    return df_return


# 用文件读数据

# In[190]:


#读取数据并将日期设置为索引（下面从文件中读取数据）
df=pd.read_excel('中证1000指数_20230717_130421.xlsx')
#df=pd.read_excel('上证50指数_20230717_151032.xlsx')
df.columns=['date','标的指数收盘价']
df.index=df['date']
df_pbpe=pd.read_excel('中证1000-历史PE／PB.xlsx')
#df_pbpe=pd.read_excel('上证50-历史PE／PB.xlsx')
df_pbpe.columns=['交易日期','PE','PB']
df_pbpe.index=df_pbpe['交易日期']
df_pbpe=df_pbpe[['PB','PE']]


# 用同花顺提取数据

# In[13]:


#用同花顺读取数据
#from iFinDPy import *
#def get_stock_data():
 #   THS_login = THS_iFinDLogin('XMDX165', '514901')
 #   if THS_login == 0:
 #       print('登录成功')
 #   else:
 #       print('登录失败')
#get_stock_data()
#用同花顺读取数据（续）
import pandas as pd
from iFinDPy import THSData
#df=THS_HQ('000852.SH','close','','2018-07-17','2023-07-17')
#df_pbpe=THS_HQ('000852.SH','pb_mrq,pe_ttm_index','','2018-07-17','2023-07-17')


# In[14]:


#对同花顺读取得到的数据进行处理
##df=THS_HQ('000852.SH','close','','2018-07-17','2023-07-17')
#df=pd.DataFrame(df)
#df.columns=['date','thscode','标的指数收盘价']
#df=df.sort_values('date')
#df.index=df['date']
#df=df[['date','标的指数收盘价']]
##df_pbpe=THS_HQ('000852.SH','pb_mrq,pe_ttm_index','','2018-07-17','2023-07-17')
#df_pbpe=pd.DataFrame(df_pbpe)
#df_pbpe.columns=['交易日期','thscode','PB','PE']
#df_pbpe=df_pbpe.sort_values('交易日期')
#df_pbpe.index=df_pbpe['交易日期']
#df_pbpe=df_pbpe[['PB','PE']]


# In[40]:


#用wind提取数据
#导入库
#from WindPy import w
#import pandas as pd
#w.start()
#数据提取
#wdata=w.wsd("000852.SH", "close", "2013-01-01", "2023-07-17", "")
#df=pd.DataFrame(wdata.Data,columns=wdata.Times,index=wdata.Codes).T
#df['date']=df.index
#df['标的指数收盘价']=df.iloc[:,0]
#df=df[['date','标的指数收盘价']]
#提取PB和PE的数据
#wdata=w.wsd("000852.SH", "pe_ttm,pb_mrq", "2013-01-01", "2023-07-17", "")
#df_pbpe1=pd.DataFrame(wdata.Times)
# df_pbpe2=pd.DataFrame(wdata.Data).T
# df_pbpe=pd.concat([df_pbpe1, df_pbpe2], axis=1)
# df_pbpe.columns=['交易日期','PE','PB']
# df_pbpe.index=df_pbpe['交易日期']
# df_pbpe=df_pbpe[['PB','PE']]


# In[191]:


df_return=snowball_dual(df,24,3,1.03,0.055,df_pbpe,"2015-01-01","2023-07-14","2023-04-14",252*3)


# In[192]:


#对PE和PB进行填充
df_return


# In[193]:


df_return.to_excel('二元小雪球回测表.xlsx')


# In[194]:


#设计一个函数计算全样本胜率
def whole_win(data):
    #全样本个数
    num_total=len(data)
    #定义条件
    #1.存续，2.未敲出
    condition1=data['产品当前是否存续']==1
    condition2=data['是否敲出']==0
    #print(condition1&condition2)
    #存续且未敲出个数
    num1=data[condition1 & condition2].shape[0]
    #已敲出或到期个数
    num2=num_total-num1
    #已敲出个数
    num3=num_total-data[condition2].shape[0]
    #全样本敲出胜率
    wrate=num3/num_total
    re=[{'全样本个数':num_total},{'存续且未敲出个数':num1},{'已敲出或到期个数':num2},{'已敲出个数':num3},{'全样本敲出胜率':wrate}]
    return re


# In[195]:


#显示全样本胜率
import pandas as pd
import numpy as np
data=pd.read_excel('二元小雪球回测表.xlsx')
whole_win(data)


# In[173]:


#生成一个PB-PE分位数随敲出胜率变化的函数，step表示步长，upper代表上界,lower代表下界,kind代表想要筛选的是PB还是PE的区间
def ana_pape(step,upper,lower,data,kind):
    upperline=upper+step
    df = pd.DataFrame({'分位数': [round(x, 2) for x in list(np.arange(lower, upperline, step))]})
    #生成一列储存胜率
    df['敲出胜率']=0
    for i in range(len(df)):
        #计算满足条件的总样本
        condition1=data['%s分位数'%kind]<=df.iloc[i,0]
        num1=data[condition1].shape[0]
        #求和的话，就写成：total = df.loc[condition1 & condition2, 'C'].sum()
        #计算满足条件且敲出的样本
        condition2=data['是否敲出']==1
        num2=data[condition1 & condition2].shape[0]
        #计算胜率并存入
        df.iloc[i,1]=num2/num1
    #df=df.style.format("{:.2%}")
    return df


# In[174]:


#读取数据并采用函数计算结果和绘图
import pandas as pd
import numpy as np
data=pd.read_excel('二元小雪球回测表.xlsx')
pb=ana_pape(0.01,1.00,0.01,data,'PB')
pe=ana_pape(0.01,1.00,0.01,data,'PE')
pbpe=pd.merge(pb,pe,on='分位数',how='outer')
pbpe.columns=['分位数','PB','PE']


# In[175]:


#绘制二维折线图
import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker

c = (
    Line()
    .add_xaxis(xaxis_data=np.sort(pbpe['分位数'].tolist()))
    .add_yaxis("PE分位数敲出胜率", y_axis=pbpe['PE'].tolist())
    .add_yaxis("PB分位数敲出胜率", y_axis=pbpe['PB'].tolist())
    .set_series_opts(label_opts=opts.LabelOpts(is_show=False))  # 不显示标签
    .set_global_opts(title_opts=opts.TitleOpts(title="分位数敲出胜率展示图"))
    .render("分位数敲出胜率展示图.html")
)


# In[177]:


#调整区间计算(kind可填'PB'/'PE')
kind='PB'
condition1=data['%s分位数'%kind]<=0.50
condition2=data['%s分位数'%kind]>=0.10
condition3=data['是否敲出']==1
winrate=data[condition1 & condition2 & condition3].shape[0]/data[condition1 & condition2].shape[0]
winrate

