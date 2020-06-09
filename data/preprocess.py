# -*- coding: utf-8 -*-
# import sys
# reload(sys)
# sys.setdefaultencoding('utf8')
import random
import time
import json
import numpy as np
import time
from datetime import datetime
import csv


# unique,label,userid,user_item_clk_1m,user_item_clk_6m,user_item_clk_1y,user_item_clk_3y,user_activation_level,item_id,brand_id,price_level,leaf_cat

def ubb_feature(userid, rec_item_info, partition, item_dict, user_clk_dict, item_filter_list, neg_cnt, fo):

    print 'userid:', userid
    print rec_item_info
    if partition == "musical_instruments":
        time_split = [14, 92, 365]
    elif partition == "electronics":
        time_split = [31, 183, 365]
    vec = rec_item_info
    # vec = vec_item_time.split(',')

    if len(vec) > 1:
        for i in range(1, len(vec)):
            itemi = vec[i].split(':')[1]
            timei = float(vec[i].split(':')[0])
            ratingi = float(vec[i].split(':')[2])

            timei_local = time.localtime(timei)
            timei_dt = time.strftime("%Y-%m-%d %H:%M:%S", timei_local)
            datei = map(int, timei_dt.split(' ')[0].split('-') + timei_dt.split(' ')[1].split(':'))
            datei = datetime(datei[0], datei[1], datei[2], datei[3], datei[4], datei[5])

            if ratingi > 3:
                l = 1
            else:
                l = 0
            labels = str(l) + ';0;0;' + vec[i].split(':')[2]

            user_items_times = []
            user_days = 0

            user_item_clk_1m = {}
            user_item_clk_6m = {}
            user_item_clk_1y = {}
            user_item_clk_3y = {}

            for j in range(i):
                itemj = vec[j].split(':')[1]
                timej = float(vec[j].split(':')[0])
                ratingj = float(vec[j].split(':')[2])
                timej_local = time.localtime(timej)
                timej_dt = time.strftime("%Y-%m-%d %H:%M:%S", timej_local)
                datej = map(int, timej_dt.split(' ')[0].split('-') + timej_dt.split(' ')[1].split(':'))
                datej = datetime(datej[0], datej[1], datej[2], datej[3], datej[4], datej[5])

                gap_time = (datei - datej).days

                if gap_time > 0:
                    if ratingj > 3:
                        user_days += 1
                        if gap_time <= time_split[0]:
                            if not user_item_clk_1m.get(itemj):
                                user_item_clk_1m[itemj] = 1
                                user_item_clk_6m[itemj] = 1
                                user_item_clk_1y[itemj] = 1
                                user_item_clk_3y[itemj] = 1
                            else:
                                user_item_clk_1m[itemj] += 1
                                user_item_clk_6m[itemj] += 1
                                user_item_clk_1y[itemj] += 1
                                user_item_clk_3y[itemj] += 1
                        elif gap_time > time_split[0] and gap_time <= time_split[1]:
                            if not user_item_clk_6m.get(itemj):
                                user_item_clk_6m[itemj] = 1
                                user_item_clk_1y[itemj] = 1
                                user_item_clk_3y[itemj] = 1
                            else:
                                user_item_clk_6m[itemj] += 1
                                user_item_clk_1y[itemj] += 1
                                user_item_clk_3y[itemj] += 1
                        elif gap_time > time_split[1] and gap_time <= time_split[2]:
                            if not user_item_clk_1y.get(itemj):
                                user_item_clk_1y[itemj] = 1
                                user_item_clk_3y[itemj] = 1
                            else:
                                user_item_clk_1y[itemj] += 1
                                user_item_clk_3y[itemj] += 1
                        else:
                            if not user_item_clk_3y.get(itemj):
                                user_item_clk_3y[itemj] = 1
                            else:
                                user_item_clk_3y[itemj] += 1


            user_item_clk_1m = sorted(user_item_clk_1m.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            user_item_clk_6m = sorted(user_item_clk_6m.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            user_item_clk_1y = sorted(user_item_clk_1y.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)
            user_item_clk_3y = sorted(user_item_clk_3y.items(), lambda x, y: cmp(x[1], y[1]), reverse=True)

            user_item_clk_1m = user_item_clk_1m[:20]
            user_item_clk_6m = user_item_clk_6m[:20]
            user_item_clk_1y = user_item_clk_1y[:20]
            user_item_clk_3y = user_item_clk_3y[:20]

            features = [userid]

            if len(user_item_clk_1m) == 0:
                features.append('0' + chr(6) + '0')
            else:
                list_f = []
                for key, value in user_item_clk_1m:
                    list_f.append(key + chr(6) + str(value))
                features.append(chr(5).join(list_f))

            if len(user_item_clk_6m) == 0:
                features.append('0' + chr(6) + '0')
            else:
                list_f = []
                for key, value in user_item_clk_6m:
                    list_f.append(key + chr(6) + str(value))
                features.append(chr(5).join(list_f))

            if len(user_item_clk_1y) == 0:
                features.append('0' + chr(6) + '0')
            else:
                list_f = []
                for key, value in user_item_clk_1y:
                    list_f.append(key + chr(6) + str(value))
                features.append(chr(5).join(list_f))

            if len(user_item_clk_3y) == 0:
                features.append('0' + chr(6) + '0')
            else:
                list_f = []
                for key, value in user_item_clk_3y:
                    list_f.append(key + chr(6) + str(value))
                features.append(chr(5).join(list_f))


            if user_days <= 5:
                user_activation_level = user_days
            else:
                user_activation_level = 6
            features.append(str(user_activation_level))

            if user_days >= 1:
                item_fea = item_dict[itemi]
                features_item = [itemi, item_fea[1], item_fea[2].replace('$', ''), item_fea[3]]
                fo.writerow([' #' + userid + '#' + itemi + '#' + vec[i].split(':')[0], labels] + features + features_item)

                if l == 1:
                    asin_neg_index_list = []
                    while (len(set(asin_neg_index_list)) < neg_cnt):
                        asin_neg_index = random.choice(item_filter_list)
                        item_neg = asin_neg_index
                        if item_neg not in user_clk_dict[userid] and item_neg != itemi:
                            asin_neg_index_list.append(asin_neg_index)
                            item_fea = item_dict[item_neg]
                            features_item_neg = [item_neg, item_fea[1], item_fea[2].replace('$', ''), item_fea[3]]

                            fo.writerow([' #' + userid + '#' + item_neg + '#' + vec[i].split(':')[0], '0;0;0;0'] + features + features_item_neg)


def preprocess_data(file1, file2):
    item_dict = {}
    item_list = []
    for line in open(file1, 'r'):
        obj = json.loads(line)
        if obj.get("title"):
            title = obj["title"].replace("\n", ' ')
        else:
            title = ''
        if obj.get("title") and obj.get("price") and obj.get("brand") and obj.get("category") and title.count(' ')<=20:
            item_dict[obj["asin"]] = [title,obj["brand"].replace("\n", ' '),obj["price"].replace("\n", ' '),obj["category"][-1].replace("\n", ' ')]
            item_list.append(obj["asin"])
    print 'item num',len(item_list)

    user_dict = {}
    user_clk_dict = {}
    item_clk_dict = {}
    f = 0
    with open(file2, 'r') as fi:
        for line in fi.readlines():
            obj = json.loads(line)
            userID = obj["reviewerID"]
            itemID = obj["asin"]
            rating = obj["overall"]

            if itemID in item_list:
                if user_clk_dict.get(userID):
                    if int(rating) > 3:
                        user_clk_dict[userID].append(itemID)
                else:
                    if int(rating) > 3:
                        user_clk_dict[userID] = [itemID]

                if item_clk_dict.get(itemID):
                    if int(rating) > 3:
                        item_clk_dict[itemID].append(userID)
                else:
                    if int(rating) > 3:
                        item_clk_dict[itemID] = [userID]
            f+=1
            # if f > 20000:
            #     break

    item_filter_list = []
    for item in item_clk_dict:
        users = item_clk_dict[item]
        if len(set(users)) > 5:
            item_filter_list.append(item)
    print 'item_filter_list', len(item_filter_list)

    user_filter_list = []
    for user in user_clk_dict:
        items = set(user_clk_dict[user])&set(item_filter_list)
        if len(items) > 1:
            user_filter_list.append(user)
    print 'user_filter_list', len(user_filter_list)

    f = 0
    with open(file2, 'r') as fi:
        for line in fi.readlines():
            obj = json.loads(line)
            userID = obj["reviewerID"]
            itemID = obj["asin"]
            rating = obj["overall"]
            time = obj["unixReviewTime"]

            if userID in user_filter_list and itemID in item_filter_list:
                if user_dict.get(userID):
                    user_dict[userID].append(str(time) + ":" + itemID + ":" + str(rating))
                else:
                    user_dict[userID] = [str(time) + ":" + itemID + ":" + str(rating)]

            f+=1
            # if f > 20000:
            #     break

    csvfile = open("./data_musical_instruments_info.csv", "w")
    fo = csv.writer(csvfile)
    for user in user_dict:
        vecs = sorted(user_dict[user])
        ubb_feature(user, vecs, 'musical_instruments', item_dict, user_clk_dict, item_filter_list, 5, fo)
    csvfile.close()

def split_test_train(flie_data):
    csvFile = open(flie_data, "r")
    reader = csv.reader(csvFile)

    # 建立空字典
    user_list = []
    result = {}
    for vec in reader:
        user_list.append(vec[2])
    user_list = set(user_list)

    test_user = random.sample(user_list,int(0.2*len(user_list)))
    train_user = list(set(user_list) - set(test_user))

    csvfile_train = open("./data_musical_instruments_info_train.csv", "w")
    fo_train = csv.writer(csvfile_train)
    csvfile_test = open("./data_musical_instruments_info_test.csv", "w")
    fo_test = csv.writer(csvfile_test)

    reader1 = csv.reader(open(flie_data, "r"))
    for feature in reader1:
        if feature[2] in train_user:
            fo_train.writerow(feature)
        else:
            fo_test.writerow(feature)

preprocess_data('./meta_Musical_Instruments.json','./Musical_Instruments.json')
split_test_train("./data_musical_instruments_info.csv")