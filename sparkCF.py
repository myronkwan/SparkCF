#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 27 22:18:25 2019

@author: myron
"""
from pyspark.mllib.recommendation import ALS
from pyspark import SparkContext,SparkConf
import sys
import csv
import math
import json
#import statistics
#import time

#%%
def main():
    assert(len(sys.argv)==6)
    train_file_name=sys.argv[1]
    bus_feature_file_name=sys.argv[2]
    test_file_name=sys.argv[3]
    case_id=int(sys.argv[4])
    output_file_name=sys.argv[5]
    def get_features(x):
        info=json.loads(x)
        return (info['business_id'],(info['stars'],info['review_count']))
    def writetofile(predictions,output_file_name):
        with open(output_file_name, 'w+') as f:
            write=csv.writer(f)
            write.writerow(['user_id','business_id','prediction'])
            write.writerows(predictions)
    
    def create_dicts(train_data,val_data):
        #get all unique user_id and bus_id from train_data
        user_train_data=train_data.map(lambda x:x[0])
        train_userids=user_train_data.collect()
        
        bus_train_data=train_data.map(lambda x: x[1])
        train_busids=bus_train_data.collect()
        #get all unique user_id and business_id from val data
        user_val_data=val_data.map(lambda x:x[0])
        val_userids=user_val_data.collect()
        
        bus_val_data=val_data.map(lambda x:x[1])
        val_busids=bus_val_data.collect()
            
        unique_userid=set(train_userids+val_userids)
        unique_busid=set(train_busids+val_busids)
    
        #create a unique_id : int mapping
        userid_int={v:k for k,v in enumerate(unique_userid)}
        int_userid={k:v for k,v in enumerate(unique_userid)}
        busid_int={v:k for k,v in enumerate(unique_busid)}
        int_busid={k:v for k,v in enumerate(unique_busid)}
        return userid_int,int_userid,busid_int,int_busid
    
    def rmse(y_true,y_pred):
        ytrue_ypred=y_true.join(y_pred) # ((user, business),(ytrue,ypred))
        rmse_val=ytrue_ypred.map(lambda x: math.sqrt((x[1][0]-x[1][1])**2)).mean()
        return rmse_val
    def pearson_corr(active_user,neighbor,corated,item_ru_ratings_dict,case):
        #return the perason correlation for active user and neighbor
        #get co-rated items between active_user and neighbor
        if case==2: #userbased
            r_avg_active=sum([item_ru_ratings_dict[(active_user,corated_bus)] for corated_bus in corated])/len(corated)
            r_neighbor_avg=sum([item_ru_ratings_dict[(neighbor,corated_bus)] for corated_bus in corated])/len(corated)
            numerator=sum([(item_ru_ratings_dict[(active_user,corated_bus)]-r_avg_active)*(item_ru_ratings_dict[(neighbor,corated_bus)]-r_neighbor_avg) for corated_bus in corated])
            if numerator==0:#if numerator is 0, denom is garaunteed to be 0
                return 0
            denom1=sum([(item_ru_ratings_dict[(active_user,corated_bus)]-r_avg_active)**2 for corated_bus in corated])
            denom2=sum([(item_ru_ratings_dict[(neighbor,corated_bus)]-r_neighbor_avg)**2 for corated_bus in corated])
            denom1=math.sqrt(denom1)
            denom2=math.sqrt(denom2)
            return numerator/(denom1*denom2)
        else:#itembased
            r_avg_active=sum([item_ru_ratings_dict[(couser,active_user)] for couser in corated])/len(corated)
            r_neighbor_avg=sum([item_ru_ratings_dict[(couser,neighbor)] for couser in corated])/len(corated)
            numerator=sum([(item_ru_ratings_dict[(couser,active_user)]-r_avg_active)*(item_ru_ratings_dict[(couser,neighbor)]-r_neighbor_avg) for couser in corated])
            if numerator==0:#if numerator is 0, denom is garaunteed to be 0
                return 0
            denom1=sum([(item_ru_ratings_dict[(couser,active_user)]-r_avg_active)**2 for couser in corated])
            denom2=sum([(item_ru_ratings_dict[(couser,neighbor)]-r_neighbor_avg)**2 for couser in corated])
            denom1=math.sqrt(denom1)
            denom2=math.sqrt(denom2)
            return numerator/(denom1*denom2)
            
        
        
    #%%

    conf=SparkConf().setAppName('hw3_task2').setMaster('local[*]')
    sc=SparkContext(conf=conf)
    #start=time.time()
    rdd=sc.textFile(train_file_name)
    rdd2=sc.textFile(test_file_name)
    #user id, bus_id, rating 
    #%%
    #remove header
    header=rdd.first()
    rdd=rdd.filter(lambda x : x!=header)
    train_data=rdd.map(lambda x: x.split(','))
    #data is now split into userid,busid,rating strings
    #do the same with validation file
    header=rdd2.first()
    rdd2=rdd2.filter(lambda x: x!=header)
    val_data=rdd2.map(lambda x:x.split(','))
    
    #%%
    if case_id==1:#case 1: model-based CF
        #Ratings object is a tuple of (int,int,float) so we must convert id's to ints with dict
        userid_int,int_userid,busid_int,int_busid=create_dicts(train_data,val_data)
        train_ratings=train_data.map(lambda c :( userid_int[c[0]],busid_int[c[1]],float(c[2])))
        val_ratings=val_data.map(lambda c : (userid_int[c[0]],busid_int[c[1]],float(c[2])))
        
        #als model
        rank=4
        numIterations=10
        lambdas=0.2
        
        model=ALS.train(train_ratings,rank,numIterations,lambdas)
        valtest=val_ratings.map(lambda x: (x[0],x[1])) #only get the userid and busid for the validation test
        predictions=model.predictAll(valtest).map(lambda x: [int_userid[x[0]],int_busid[x[1]],x[2]]).collect()
        writetofile(predictions,output_file_name)

        #predictions=model.predictAll(valtest).map(lambda x: ((x[0],x[1]),x[2]))
        #ytrue=val_ratings.map(lambda x: ((x[0],x[1]),x[2]))
    #print(rmse(ytrue,predictions))
    #end=time.time()
    #print(end-start)
    
    #%%    
    elif case_id==2:#case 2: user-based
    #create a dict of user and all businesses they reviewed
        user_bus=train_data.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set).collectAsMap()
        #bus_user: dict of business: users set
        bus_user=train_data.map(lambda x: (x[1],x[0])).groupByKey().mapValues(set).collectAsMap()
        #dict (user,bus), rating
        train_ratings=train_data.map(lambda x: ((x[0],x[1]),float(x[2]))).collectAsMap()
        #dict for saving avg of a user for calculations
        avg_user_rating={}
        predictions=[]
        #rsme_=[]
        
        
        #calculate avg rating for all users for all businesses for prediction where we have never seen the user or the rating
        allratings=train_data.collect()
        global_avg=sum([float(v[2]) for v in allratings])/len(allratings)
        del allratings
        threshold_corated=0.3
        for a_user,a_bus,rating in val_data.toLocalIterator():
            #if we have this user and business in our training dat
            if a_user in user_bus and a_bus in bus_user:
                #calculate the avg rating of this user if we havent yet
                if a_user not in avg_user_rating:
                    #get ratings of all business active user has rated
                    all_ratings=[train_ratings[(a_user,business)] for business in user_bus[a_user]]
                    #save tuple of (sum of ratings, total num of ratings made). if we want avg, need to divide 
                    avg_user_rating[a_user]=(sum(all_ratings),len(all_ratings))
                avg_rating=avg_user_rating[a_user][0]/avg_user_rating[a_user][1]
                #all neighbors are other users that have also rated the business
                neighbors=bus_user[a_bus]
                
                #get pearson correlation of all these neighbors with the pearson corr algorithm
                w=[] #list of (neighbor, pearson corr)
                for neighbor in neighbors:
                    corated=user_bus[a_user].intersection(user_bus[neighbor])
                    if len(corated)/len(user_bus[a_user].union(user_bus[neighbor]))>=threshold_corated:#add neighbor do dict if num corated items >threshold
                    #if len(corated)>=threshold_corated:#add neighbor do dict if num corated items >threshold
                        pearson=pearson_corr(a_user,neighbor,corated,train_ratings,2)
                        if pearson>0:
                            w.append((neighbor,pearson))
                #make weighted prediction for a_user
                #make sure we have avg_user_rating of all relevant neighbors
                numerator=0
                denom=0
                for user,wau in w:#user is neighbor
                    if user not in avg_user_rating:
                        all_ratings=[train_ratings[(user,business)] for business in user_bus[user]]
                        avg_user_rating[user]=(sum(all_ratings),len(all_ratings))
                    
                    if avg_user_rating[user][1]>1:
                        avg_neighbor=(avg_user_rating[user][0]-train_ratings[(user,a_bus)])/(avg_user_rating[user][1]-1)
                    else:
                        avg_neighbor=0
                    numerator+=((train_ratings[(user,a_bus)]-avg_neighbor)*wau)
                    denom+=abs(wau)
                    
                if denom==0: #if no close users
                    prediction=avg_rating#just append avg rating of this user
                else:
                    prediction=avg_rating+(numerator/denom)
        
            elif a_user in user_bus:#if we have never seen this business
                #average rating of this user for this business
                prediction=sum([train_ratings[(a_user,bus)] for bus in user_bus[a_user]])/len(user_bus[a_user])
            elif a_bus in bus_user:#if we have never seen this user
                #average rating of other this business from all other users 
                prediction=sum([train_ratings(user,a_bus) for user in bus_user[a_bus]])/len(bus_user[a_bus])
            else:#if we have never seen either user or business
                prediction=global_avg
            predictions.append([a_user,a_bus,prediction])
            #rsme_.append((float(rating)-prediction)**2)
        writetofile(predictions,output_file_name)
        #end=time.time()
        #print(math.sqrt(statistics.mean(rsme_)))
        #print(end-start)
    
    #%%
    elif case_id==3:#3: item-based
        user_bus=train_data.map(lambda x: (x[0],x[1])).groupByKey().mapValues(set).collectAsMap()
        #bus_user: dict of business: users set
        bus_user=train_data.map(lambda x: (x[1],x[0])).groupByKey().mapValues(set).collectAsMap()
        #dict (user,bus), rating
        train_ratings=train_data.map(lambda x: ((x[0],x[1]),float(x[2]))).collectAsMap()
        #dict for saving avg of a user for calculations
        avg_bus_rating={}
        predictions=[]
        #error=[]
        allratings=train_data.collect()
        global_avg=sum([float(v[2]) for v in allratings])/len(allratings)
        for a_user,a_bus,rating in val_data.toLocalIterator():
            #if we have this user and business in our training dat
            if a_user in user_bus and a_bus in bus_user:
                if a_bus not in avg_bus_rating:
                    all_ratings=[train_ratings[(user,a_bus)] for user in bus_user[a_bus]]
                    avg_bus_rating[a_bus]=(sum(all_ratings),len(all_ratings))
                avg_rating=avg_bus_rating[a_bus][0]/avg_bus_rating[a_bus][1]
        
                neighbors=user_bus[a_user]#all other businesses that this user has rated
        
                w=[]
                threshold_corated=0.3 #threshold for similar item
                for neighbor in neighbors:
                    corated=bus_user[a_bus].intersection(bus_user[neighbor])
                    if len(corated)/len(bus_user[a_bus].union(bus_user[neighbor]))>=threshold_corated:
                        pearson=pearson_corr(a_bus,neighbor,corated,train_ratings,3)
                        if pearson>0:
                            w.append((neighbor,pearson))
                for bus,wau in w:
                    if bus not in avg_bus_rating:
                        all_ratings=[train_ratings[(user,bus)] for user in bus_user[bus]]
                        avg_bus_rating[bus]=(sum(all_ratings),len(all_ratings))
                numerator=sum([train_ratings[(a_user,bus)]*wau for bus,wau in w])
                denom=sum([abs(wau[1]) for wau in w])
                if denom==0:
                    prediction=avg_rating
                else:
                    prediction=numerator/denom
        
            elif a_user in user_bus:#if we have never seen this business
                #average rating of this user for this business
                prediction=sum([train_ratings[(a_user,bus)] for bus in user_bus[a_user]])/len(user_bus[a_user])
            elif a_bus in bus_user:#if we have never seen this user
                #average rating of other this business from all other users 
                prediction=sum([train_ratings(user,a_bus) for user in bus_user[a_bus]])/len(bus_user[a_bus])
            else:
                prediction=global_avg
            predictions.append([a_user,a_bus,prediction])
            #error.append((float(rating)-prediction)**2)
        writetofile(predictions,output_file_name)
        #end=time.time()

        #print(end-start)
    
        
if __name__=='__main__':
    main()
