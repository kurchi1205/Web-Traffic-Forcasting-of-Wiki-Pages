def final(page,start_date,end_date):
    import pandas as pd
    from datetime import datetime,timedelta
    import numpy as np
    from tensorflow.keras.initializers import glorot_uniform
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras import backend as K
    from tensorflow.keras.models import load_model
    from tensorflow.keras.utils import CustomObjectScope

    

    

        
    import re
    start_date=datetime.strptime(start_date, '%Y-%m-%d')
    end_date=datetime.strptime(end_date, '%Y-%m-%d')
    delta = end_date-start_date
    days= delta.days+1
    new_date = end_date + timedelta(10)
    datelist = pd.date_range(start_date,new_date-timedelta(days=1),freq='d')   
    weekday_test=[]
    print("Creating the features ...  ")
    for i in datelist:
        weekday_test.append(i.weekday())
    weekday_test=pd.Series(weekday_test)

    month_test=[]
    for i in datelist:
        month_test.append(i.month)
    month_test=pd.Series(month_test)

    month_start_test=[]
    month_start_test=pd.Series(datelist).dt.is_month_start

    month_end_test=[]
    month_end_test=pd.Series(datelist).dt.is_month_end

    quarter_start_test=[]
    quarter_start_test=pd.Series(datelist).dt.is_quarter_start

    quarter_end_test=[]
    quarter_end_test=pd.Series(datelist).dt.is_quarter_end

    week_test=[]
    week_test=pd.Series(datelist).dt.week


    quarter_test=[]
    quarter_test=pd.Series(datelist).dt.quarter


    days_in_month_test=[]
    days_in_month_test =pd.Series(datelist).dt.days_in_month

    year_test=[]
    year_test=pd.Series(datelist).dt.year


    is_sunday_or_monday_test=[]
    for i in weekday_test:
        if i == 0 or i == 6:
            is_sunday_or_monday_test.append(1)
        else:
            is_sunday_or_monday_test.append(0)
    is_sunday_or_monday_test=pd.Series(is_sunday_or_monday_test)


    is_august_test=[]
    for i in month_test:
        if i == 8:
            is_august_test.append(1)
        else:
            is_august_test.append(0)
    is_august_test=pd.Series(is_august_test)

    year_half_test=[]
    for i in month_test:
        if i in [1,2,3,4,5,6] :
            year_half_test.append(1)
        else :
            year_half_test.append(2)
    year_half_test=pd.Series(year_half_test)

### The above features are irrespective of the page , I will call them global features
    global_feat=pd.DataFrame()
    global_feat=pd.concat([weekday_test,is_sunday_or_monday_test,month_test,is_august_test,year_half_test,quarter_test,quarter_start_test,quarter_end_test,month_start_test,month_end_test,days_in_month_test,week_test],axis=1)
    global_feat.columns=['weekday','is_sunday_or_monday','month','is_august','year_half','quarter','quarter_start','quarter_end','month_start','month_end','days_in_month','week']
    
    def access(page):
        all_access=re.search('all-access',page)
        desktop=re.search('desktop',page)   
        mobile=re.search('mobile-web',page)
        if(all_access):    
            return (0)
        elif(desktop):
            return(1)
        else:
            return(2)
    def agent(page):
        index=re.search('spider',page)
   
        if(index):
            return (1)
        else:
            return(0)

    access_index=access(page)
    agent_index=agent(page)
    pageview = np.load('viewperc.npy',allow_pickle='TRUE').item()
    viewperc= pageview[page]

    viewmid=pd.read_csv('viewmid.csv')
    
    view1=viewmid.loc[0].values[0]
    view2=viewmid.loc[1].values[0]
    view3=viewmid.loc[2].values[0]
    view4=viewmid.loc[3].values[0]
    
    if agent_index == 1:
        spider=[1]*(global_feat.shape[0])
        non_spider=[0]*(global_feat.shape[0])
    else:
        spider=[0]*(global_feat.shape[0])
        non_spider=[1]*(global_feat.shape[0])
    spider=pd.Series(spider)
    non_spider=pd.Series(non_spider)
    page_specific_feat=pd.DataFrame()
    page_specific_feat=pd.concat([spider,non_spider],axis=1)
    page_specific_feat.columns=['spider','non_spider']
    if access_index==0:
        page_specific_feat['All_Access']=1
        page_specific_feat['Desktop']=0
        page_specific_feat['Mobile']=0

    elif access_index==1:
        page_specific_feat['All_Access']=0
        page_specific_feat['Desktop']=1
        page_specific_feat['Mobile']=0
    else:
        page_specific_feat['All_Access']=0
        page_specific_feat['Desktop']=0
        page_specific_feat['Mobile']=1

    total_feat=pd.concat([global_feat,page_specific_feat],axis=1)
    
    print("Feature Created ...")
    
    print("Preprocessing the data ... ")
    from sklearn.preprocessing import LabelEncoder
    le1=LabelEncoder()
    total_feat['month_start']=le1.fit_transform(total_feat['month_start'])


    le2=LabelEncoder()
    total_feat['month_end']=le2.fit_transform(total_feat['month_end'])


    le3=LabelEncoder()
    total_feat['quarter_start']=le3.fit_transform(total_feat['quarter_start'])


    le4=LabelEncoder()
    total_feat['quarter_end']=le4.fit_transform(total_feat['quarter_end'])



    def create_test_dataset(X,timestep=1):
        Xs=[]
        for i in range(len(X)):
            end_ix=i+timestep
            if end_ix > X.shape[0]:
                break
            
            v=X[i:end_ix]
            Xs.append(v)
            
        return np.array(Xs)

    total_feat=np.log1p(total_feat)
    test_x=create_test_dataset(total_feat.values,7)
    
    print("Preprocessing Completed ... ")
    
    
    def customLoss(y_true, y_pred):
        epsilon = 0.1
        summ = K.maximum(K.abs(y_true) + K.abs(y_pred) + epsilon, 0.5 + epsilon)
        smape = K.abs(y_pred - y_true) / summ * 2.0
        return smape

    opt=Adam(learning_rate=0.001)

    with CustomObjectScope({'GlorotUniform': glorot_uniform()}):
        model1=load_model('model1_new.hdf5',compile=False)
        model1.compile(loss=customLoss,optimizer=opt)
        model2=load_model('model2_new.hdf5',compile=False)
        model2.compile(loss=customLoss,optimizer=opt)
        model3=load_model('model3_new.hdf5',compile=False)
        model3.compile(loss=customLoss,optimizer=opt)
        model4=load_model('model4_new.hdf5',compile=False)
        model4.compile(loss=customLoss,optimizer=opt)
        model5=load_model('model5_new.hdf5',compile=False)
        model5.compile(loss=customLoss,optimizer=opt)
        model6=load_model('model6_new.hdf5',compile=False)
        model6.compile(loss=customLoss,optimizer=opt)
        model7=load_model('model7_new.hdf5',compile=False)
        model7.compile(loss=customLoss,optimizer=opt)
        model8=load_model('model8_new.hdf5',compile=False)
        model8.compile(loss=customLoss,optimizer=opt)
 
    print("Predicting the pagehits ... ")
    if access_index==0 and agent_index==0:
        if viewperc>=view1:
            y_pred_lstm=model1.predict(test_x)
        else:
            y_pred_lstm=model5.predict(test_x)

    elif access_index==1 and agent_index==0:
        if viewperc>=view2:
            y_pred_lstm=model2.predict(test_x)
        else:
            y_pred_lstm=model6.predict(test_x)

    elif access_index==2 and agent_index==0:
        if viewperc>=view3:
            y_pred_lstm=model3.predict(test_x)
        else:
            y_pred_lstm=model7.predict(test_x)
    elif access_index==0 and agent_index==1:
        if viewperc>=view4:
            y_pred_lstm=model4.predict(test_x)
        else:
            y_pred_lstm=model8.predict(test_x)
    
    y_pred_lstm=y_pred_lstm[0:days]
    y_pred_lstm=np.exp(y_pred_lstm)-1    
    y_pred_lstm=pd.DataFrame(y_pred_lstm)
    y_pred_lstm.index=datelist[0:days]
    y_pred_lstm=y_pred_lstm.transpose()
    y_pred_lstm.index=[page]
    print("Task Completed ... ")
    return y_pred_lstm
    
    
