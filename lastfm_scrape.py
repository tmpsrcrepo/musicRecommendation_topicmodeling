# -*- coding: utf-8 -*-
import pylast
import urllib
import httplib
import string
import re
import csv
from itertools import islice

#need to replace your api_key, api_secret, username and password on last.fm
api_key='___'
api_secret='___'
username='___'
password_hash=pylast.md5('____')

network = pylast.LastFMNetwork(api_key = api_key, api_secret =
    api_secret, username = username, password_hash = password_hash)


list_100=network.get_top_tracks(100)
artist_100 = network.get_top_artists(100)

artist_100[10]
#get individual tracks

network.get_geo_top_artists('United States')
network.get_geo_top_tracks('United States')
#network.

alb=network.get_album('Cher','Believe')
print alb.get_listener_count()
print alb.get_playcount()
tracks=alb.get_tracks()
track1=tracks[0]
track1.get_similar()
#create the dictionary of genres - genre.csv
#lower case, replace punctuation by empty space
genre_sub={}
with open('genre_subcategory.csv','r') as f:
    for line in f:
        l = line.split(',')
        index=0
        genre=''
        for w in l:
            w=w.strip('\n').lower()
            if w!='':
                if index==0:
                    genre=w
                else:
                    genre_sub[w]=genre.lower()
                #print w
                index+=1
                
print genre_sub
            
    
wfile = csv.writer(open('albums_labels1.csv','w'))

with open ('albums - select.csv','r') as f:
    for line in f:
        l = line.split(',')
        album=l[1]
        artist=l[2]
        label=l[3]
        year=l[4]
        score=l[6]
        url=l[9]
        #get tags
        try:
            alb = network.get_album(artist,album)
            
            for t in islice(alb.get_top_tags(),10):
                item_ = str(t.item)
                if item_.__contains__('&'):
                    item_=item_.replace('&','n')
                else:
                    item_=item_.replace('-',' ')
                item_=item_.lower()
                if genre_sub.__contains__(item_):
                    print album
                    print item_
                    wfile.writerow([album,artist,label,year,score,genre_sub.get(item_),url])
                    break
        except Exception:
            pass
        #if tag!=empty, write to wfile


wfile = csv.writer(open('tags_test_set_final1.csv','w'))

with open ('tags.csv','r') as f:
    for line in f:
        l = line.rstrip('\n')
        if l!='"':
            l=l.split(',')
            album=l[0]
            artist=l[1]
            label=l[2]
            year=l[3]
            score=l[4]
            url=l[6]
            #get tags
            try:
                alb = network.get_album(artist,album)
                
                lis=[]
                for t in islice(alb.get_top_tags(),150):
                    lis.append(str(t.item))
                print lis
                
                wfile.writerow([album,artist,label,year,score,url,]+lis)
                    
            except Exception:
                pass              
#read each line, get artist and album name
#get tag 
#album=network.get_album('Babatunde Olatunji','Drums of Passion')
#for t in islice(album.get_top_tags(),10):
    #print t.item #extract items! success!
    #replace punctuation by empty space
    #replace & or and by n


        



