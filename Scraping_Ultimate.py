# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 01:07:58 2015

@author: Xiaoqian
"""
#initialization of pychonest
from pyechonest import config
from pyechonest import artist,song
import cPickle
import pylast
import urllib
import httplib
import string
import re
import csv
from itertools import islice
from topic_model import *

#initialization of pylast, 

api_key='91a81936a4e18b75ad33d62892e36e23'
api_secret='f880ead2f0243c913a050a37c5d33f1d'

username='wings_chains'
password_hash=pylast.md5('Lxq19911116')
config.ECHO_NEST_API_KEY = 'ZIK3QQ0RMMM1TOP2U'

network = pylast.LastFMNetwork(api_key = api_key, api_secret =
    api_secret, username = username, password_hash = password_hash)
#test set
#get top 10 tags, 
def put_Set(c, val,map):
    if not map.__contains__(c):
        map[c]=set()
        map[c].add(val)
    else:
        
        map[c].add(val)

        

    
class Track:
    def __init__(self,name,artist,album):
        self.name = name
        self.artist = artist
        self.album = album
        self.types = []
        self.audio_summary={}
        





def Album_attributes(toread):
    alb_songs={}
    score_songs={}
    
    #network = pylast.LastFMNetwork(api_key = api_key, api_secret =api_secret, username = username, password_hash = password_hash)
    #config.ECHO_NEST_API_KEY = 'ZIK3QQ0RMMM1TOP2U'
    with open (toread,'r') as f:
        for line in f:
            
            l = line.rstrip('\n')
            
            if l!='"' and l!=None and l!='':
                l=l.split(',')
                album=l[0]
                artist_=l[1]
                score=l[4]
                #get tags
                try:
                    print artist
                    alb = network.get_album(artist_,album)
                    
                    for i in alb.get_tracks():
                        song_ = i.get_name()
                        print "***************Track List**************************"
                        print song_
                        #add to alb_songs
                        
                        #collect song features
                        
                        song_result = song.search(artist=str(artist_),title=str(song_))
                        
                        if len(song_result)>0:
                            song_result=song_result[0]
                            print song_result
                            
                            #if song_res!=None:
                            print "*************Song Result************************"
                            
                            track_ = Track(song_,artist_,album)
                            track_.types=song_result.song_type
                            summary = song_result.audio_summary
                        
                            
                            for k,v in summary.items():
                                if k!='analysis_url' and k!='audio_md5':
                                    track_.audio_summary[k]=v
                            track_.audio_summary['hottness']=song_result.get_song_hotttnesss()
                            track_.audio_summary['currency']=song_result.get_song_currency()
                            track_.audio_summary['discovery']=song_result.get_song_discovery()
                            #may need to add artist info
                            track_.audio_summary['artist_fam']=song_result.artist_familiarity
                            track_.audio_summary['artist_hot']=song_result.artist_hotttnesss
                            print len(trsack_.audio_summary)
                            #add to score_song dictionary (score)
                            map_putlist(score,track_,score_songs) 
                            map_putlist(album,track_,alb_songs)
                            
                except Exception:
                    pass
    saveDict(alb_songs,'alb_songs1')
    saveDict(score_songs,'score_songs1')
    return alb_songs,score_songs

class Track_tag:
    def __init__(self,name,pair):
        self.name = name
        self.pair=pair
        self.tags = []
        self.similar = []
        self.match=[]

def getTags(song_,lis):
    for t in islice(song_.get_top_tags(),150):
        lis.append(str(t.item))


def getSimilar(song_,nameList,matchList):
    for t in song_.get_similar():
        nameList.append(t.item.get_name())
        matchList.append(t.match)
#alb_songs,score_songs=Album_attributes('pitchfork_all.csv')


def getTracks(genre_):
    album_list={}
    track_tag_list={}
    for k,v in genre_.items():
        for alb in v:
            artist_=alb.artist
            name_ = alb.name
            pair = name_+'-'+artist_
            try:
                album_ = network.get_album(artist_,name_)
                print name_
                for i in album_.get_tracks():
                    song_ = i.get_name()
                    print song_
                    #add tracks to the same album (assigned with the same rating)
                    put_Set(pair,(song_),album_list)
                    
                    #add tags and find similar ratings
                    track_tag = Track_tag(song_,pair)
                    getSimilar(i,track_tag.similar,track_tag.match)
                    getTags(i,track_tag.tags)
                    track_tag_list[song_]=track_tag
            except Exception:
                pass
               # print Exception
            
    return album_list,track_tag_list


