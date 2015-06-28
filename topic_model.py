# -*- coding: utf-8 -*-
"""
Created on Sat Apr 25 20:41:37 2015

@author: Xiaoqian
"""
from itertools import islice
from gensim import corpora, models, similarities,matutils
from collections import defaultdict
from bs4 import BeautifulSoup
import codecs
import glob
import cPickle
import re,string
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.decomposition import PCA
import math
#import nltk
from nltk.stem import PorterStemmer
from random import uniform
import time
import numpy as np
import pylab
import urllib2
from operator import or_
import matplotlib.pyplot as plt
from nltk.tag.stanford import NERTagger
import os
import subprocess
from sklearn import svm
import pylast
from pyechonest import config
#from pyechonest import artist,song

#Replace the following contents by yours
api_key='___'
api_secret='__'
username='___'
password_hash=pylast.md5('_____')
config.ECHO_NEST_API_KEY = '___'

network = pylast.LastFMNetwork(api_key = api_key, api_secret =
    api_secret, username = username, password_hash = password_hash)
    
    
def map_put(c,val,map):
    if not map.__contains__(c):
        map[c]=val
    else:
        map[c]+=val

def map_putlist(c, val,map):
    if not map.__contains__(c):
        map[c]=[val]
    else:
        map[c].append(val)

def updateDF(temp_map,df,c):
    if not temp_map.__contains__(c):
        temp_map[c]=1
        map_put(c,1,df)
    else:
        temp_map[c]+=1


############Tokenize tags collected from last.fm##################

def tokenizeTags(str,dict_items):
    #temp map (for getting the local term frequency)
    #for a sentence
    str =str.decode('ascii', 'ignore')
    #tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    #tokens=tokenizer.tokenize(str)
    tokens = str.split()
    #print tokens
    stemmer = PorterStemmer()
    #small set of stopwords (remove you, are, and, I those kinds of words)
    last =[]
    #bigram_list=[]
    for d in tokens:
        d = d.split('-')
        for c in d:
                c=re.compile('[%s]' % re.escape(string.punctuation)).sub('', c)
                #regular expression -> strip punctuations
                if c!='' and c not in dict_items:
                    try:
                        if int(c):
                            if len(c)!=4 and (c>2015 or c<1900): #keep years
                                c=stemmer.stem('NUM')
                    except Exception:
                        c = stemmer.stem(c.lower())
                        pass
                    #c = stemmer.stem(c.lower())
                    last.append(c)
                    #bigram generation
                #index= len(last)
                #if index>1:
                   # bigram = last[index-2]+' '+last[index-1]
                   # bigram_list.append(bigram)
    return last

def tokenize2(str,df_freq):
    #temp map (for getting the local term frequency)
    temp_map={}
    #for a sentence
    str =str.decode('ascii', 'ignore')
    #tokenizer = nltk.tokenize.treebank.TreebankWordTokenizer()
    #tokens=tokenizer.tokenize(str)
    tokens = str.split()
    #print tokens
    stemmer = PorterStemmer()
    #small set of stopwords (remove you, are, and, I those kinds of words)
    
    
    last =[]
    #bigram_list=[]
    for d in tokens:
        d = d.split('-')
        for c in d:
            c=re.compile('[%s]' % re.escape(string.punctuation)).sub('', c)
                #regular expression -> strip punctuations
                if c!='':
                    try:
                        if int(c):
                            if len(c)!=4 and (c>2015 or c<1900): #keep years
                                c=stemmer.stem('NUM')
                    except Exception:
                        c = stemmer.stem(c.lower())
                        pass
                    
                    last.append(c)
                    updateDF(temp_map,df_freq,c)
    return last

    
def tokenize2_bigram(str,df_freq):
    temp_map={}
    #for a sentence
    str =str.decode('ascii', 'ignore')
    tokens = str.split()
    #print tokens
    stemmer = PorterStemmer()
    last =[]
    bigram_list=[]
    for d in tokens:
        d = d.split('-')
        for c in d:
                c=re.compile('[%s]' % re.escape(string.punctuation)).sub('', c)
                #regular expression -> strip punctuations
                if c!='':
                    try:
                        if int(c):
                            if len(c)!=4 and (c>2015 or c<1900): #keep years
                                c=stemmer.stem('NUM')
                    except Exception:
                        c = stemmer.stem(c.lower())
                        pass
                    
                    #c = stemmer.stem(c.lower())
                    last.append(c)
                    
                    #bigram generation
                index= 0
                if index>1:
                    bigram = last[index-2]+' '+last[index-1]
                    bigram_list.append(bigram)
                    updateDF(temp_map,df_freq,bigram)
                    index+=1
    return bigram_list
    
######################End of tokenization#############################


#######################Album objects##################################
class album_simp:
    def __init__(self,name,artist,tokens,genre,score):
        self.name=name
        self.artist=artist
        self.tokens=tokens
        self.genre=genre
        self.score = score
    def __eq__(self,a):
        if self.name == a.name:
            return True

def initializeSoup(start):
    if start[0]!='h':
        start=start[1:]
    page=urllib2.urlopen(start).read()
    soup=BeautifulSoup(page)
    return soup


#######################Review Scraping from Pitchfork.com##################

def reviewScraping1(link):
    soup=initializeSoup(link)
    rev = ''
    for div in soup.find_all('div',{'class':'editorial'}):
        for para in div.find_all('p'):
            rev+=(para.get_text()).encode('ascii','ignore')
            
        for l in div.find_all('li'):
            rev+=(l.get_text()).encode('ascii','ignore')
    return rev

######################Processing the review contents#######################
def processing1(train_map,genre_map,score_map):
    artist=''
    album=''
    index = 1
    for k,v in train_map.items():
        tmp = k.split('-')
        artist = tmp[1]
        album = tmp[0]
        link = v[4]
        year= v[1]
        print link
        rev=reviewScraping1(link)
        
        rev+=('.'+year)
        index+=1
        genre = v[3]
        score = v[2]
        
        rate_interval = 0
        if float(score)<=5:
            rate_interval = 1
        else:
            if float(score)<=7:
                rate_interval = 2
            else:
                if float(score)<=8:
                    rate_interval=3
                else:
                    rate_interval = 4
        
        album_obj = album_simp(album,artist,rev,genre,score)
        map_putlist(genre,album_obj,genre_map)
        map_putlist(rate_interval,album_obj,score_map)
        
#k,v 
def saveDict(dict_,name):
    with open(name+'.pickle','wb') as f:
        cPickle.dump(dict_,f)

def loadDict(name):
    with open(name,'r') as f:
        return cPickle.load(f)        


#####################Read from a list of reviews (train set)###############
def readF(path_train):
    genre_map={}
    score_map={}
    #read through train.csv
    train_map={}
    with open(path_train,'r') as tr:
        for line in tr:
            l = line.rstrip('\n').split(',')
            if l!=['"']:
                k = l[0]+'-'+l[1]
                #label = 2, year = 3, score=4,genre=5
                train_map[k]=[l[2],l[3],l[4],l[5],l[6]]
                #scraping
    
    processing1(train_map,genre_map,score_map)
    #for f in files:
        #processing(data,train_map,genre_map,score_map)
        
        #save genre_map and score_map        
        
    return genre_map,score_map

###################Creating a dictionary object in gensim######################

def Dict_(texts,controlvocab):
    texts1=[]
    for doc in texts:
        tmp=[]
        for t in doc:
            if t in controlvocab:
                tmp.append(t)
        texts1.append(tmp)
    dictionary = corpora.Dictionary(texts1)
    corpus = [dictionary.doc2bow(text) for text in texts1]
    return dictionary,corpus
    
##################Applying LDA model after processing##########################
def ldaModel(dictionary,corpus,k_percent):
    num_topics_ = int(k_percent*dictionary.token2id.__len__())
    ldamodel = models.LdaModel(corpus,id2word=dictionary,num_topics=num_topics_,passes = 20, eval_every=5)        
    return ldamodel

def lda_genre(genres_):
    SEED = 42
    # before training/inference:
    np.random.seed(SEED)
    lda_list=[]
    dict_list=[]
    corpus_list=[]
    #alltext =[]
    for k,v in genres_.items():
        lis = []
        print k
        for alb in v:
            lis.append(alb.tokens)
            #alltext.append(alb.tokens)
        dict_tmp,corpus_tmp=Dict_(lis)
        print dict_tmp.__len__()
        dict_list.append(dict_tmp)
        corpus_list.append(corpus_tmp)
        ##lda_list.append(ldaModel(dict_tmp,corpus_tmp,0.05))
        lda_list.append(ldaModel(dict_tmp,corpus_tmp,0.008))
    
    
    for ind in range(0,len(dict_list)):
        name = genres_.keys()[ind]
        dict_list[ind].save(name+'.dict')
        corpora.MmCorpus.serialize(name+'corpus.mm', corpus_list[ind])
    return lda_list,dict_list,corpus_list#,alltext

def saveHdp(hdplist,keys):
    index=0
    for hdp in hdplist:
        hdp.save(keys[index]+'_unigram.hdp')
        index+=1
def saveHdp1(hdplist,keys):
    index=0
    for hdp in hdplist:
        hdp.save(keys[index]+'_bigram.hdp')
        index+=1
def controlVocab(df,numDoc):
    web=urllib2.urlopen("http://jmlr.org/papers/volume5/lewis04a/a11-smart-stop-list/english.stop")
    stemmer = PorterStemmer()
    for i in web:
        word=stemmer.stem(i.strip())
        word=re.compile('[%s]' % re.escape(string.punctuation)).sub('', word)
        if df.__contains__(word.lower()):
            del df[word]
    #remove low-freq
    for k,v in df.items():
        if v<5 or v>=numDoc:
            del df[k]
    return sorted(df, key=df.get,reverse=True)[100:]



    
def hdp_genreBigram(genres_):
    SEED = 42
    # before training/inference:
    np.random.seed(SEED)
    df_freq={}
    alllist={}
    numDoc=0
    for k,v in genres_.items():
        print k
        for alb in v:
            #create DF
            map_putlist(k,tokenize2_bigram(alb.tokens,df_freq),alllist)
            numDoc+=1
        #alllist[k]=lis
    print df_freq.__len__()
    controlvocab = controlVocab(df_freq,numDoc)
    print controlvocab.__len__()
    #filter out background tokens (shared in every doc)
    for k,v in alllist.items():
        dict_tmp,corpus_tmp=Dict_(v,controlvocab)
        #update the list
        del alllist[k]
        dict_tmp.save(k+'_bigram_.dict')
        corpora.MmCorpus.save(k+'_bigram.mm')
        hdp_m=models.HdpModel(corpus_tmp,dict_tmp)
        hdp_m.save(k+'_bigram.hdp')
        
    print 'finished'
    

        
    
def hdp_genre(genres_):
    SEED = 42
    # before training/inference:
    np.random.seed(SEED)
    df_freq={}
    hdp_list=[]
    dict_list=[]
    corpus_list=[]
    alllist={}
    numDoc=0
    for k,v in genres_.items():
        print k
        for alb in v:
            #create DF
            map_putlist(k,tokenize2(alb.tokens,df_freq),alllist)
            numDoc+=1
        #alllist[k]=lis
    print df_freq.__len__()
    controlvocab = controlVocab(df_freq,numDoc)
    print controlvocab.__len__()
    #filter out background tokens (shared in every doc)
    for k,v in alllist.items():
        dict_tmp,corpus_tmp=Dict_(v,controlvocab)
        print dict_tmp.__len__()
        #update the list
        dict_list.append(dict_tmp)
        corpus_list.append(corpus_tmp)
        hdp_list.append(models.HdpModel(corpus_tmp,dict_tmp))
    

    for ind in range(0,len(dict_list)):
        name = genres_.keys()[ind]
        dict_list[ind].save(name+'__.dict')
        corpora.MmCorpus.serialize(name+'__corpus.mm', corpus_list[ind])
    return hdp_list,dict_list,corpus_list#,alltext    
    

   
def sample_generate(genres_,dict_whole,lda_file,ktopics):
    labels = []
    lda_m = models.LdaModel.load(lda_file)
    output=[]
    keys=genres_.keys()
    for k,v in genres_.items():
        for alb in v:
            labels.append(keys.index(alb.genre))
            query = tokenizeTags(alb.tokens,dict_whole.itervalues())
            query = dict_whole.doc2bow(query)
            vec =  lda_m[query]
            vec_topic = ktopics*[0]
            for i in vec:
                vec_topic[i[0]]=i[1]
            #print vec_topic
            output.append(vec_topic)
    return np.array(output),np.array(labels)




#score map: return a list of albums within



def test_Corpus(docs,dictionary):
    corpus = [dictionary.doc2bow(text) for text in docs]
    tfidf = models.TfidfModel(corpus)
    #ldamodel.update(corpus)
    return corpus,tfidf

def tfidf_albums(corpus):
    tfidf = models.TfidfModel(corpus)
    return tfidf
    
    
def SVM_(sample,labels):
    clf = svm.SVC()
    clf.fit(sample,np.array(labels))


def input_Song(title,artist_):
    try:
        song_=network.get_track(artist_,title)
        return song_ 
    except Exception:
        pass

def input_Album(title,artist_):
    try:
        album_=network.get_album(artist_,title)
        return album_
    except Exception:
        pass
    
def getTags(song_,lis,artist_):
    for t in song_.get_top_tags():
        lis.append(str(t.item))    
    
    for t in network.get_artist(artist_).get_top_tags():
        lis.append(str(t.item))
#print numpy_matrix
def getTracks(name,artist_):
    track_tags={}
    try:
        album_= network.get_album(artist_,name)
        for i in album_.get_tracks():
            lis=[]
            song_=i.get_name()
            getTags(i,lis,artist_)
            track_tags[song_+'-'+artist_]=lis
    except Exception:
        pass
               # print Exception
    if track_tags.__len__()>0:
        return track_tags
            
def ApplyHDP_tag(tags1,dict_,hdpModel):
    tags = dict_.doc2bow(tags1)
    vec1 = hdpModel[tags]
    return vec1

def HDP_Wrapper(genre_,title,artist_,k):
    tags=[]
    tracks=[]
    track_=input_Song(title,artist_)
    getTags(track_,tags,artist_)
    Sim_dictionary={}
    for k_,v in genre_.items():
        dict_ = corpora.Dictionary.load(k_+'__.dict')
        corpus_=corpora.MmCorpus(k_+'__corpus.mm')
        hdp = models.HdpModel.load(k_+'_unigram.hdp')
        #tokenize tags
        tags1=[]
        for tag in tags:
            tags1+=tokenizeTags(tag,dict_.itervalues())
        #query
        vec1 = ApplyHDP_tag(tags,dict_,hdp)
        
        sim,track_genre=search_space(vec1,v,dict_,corpus_,hdp,k)
        Sim_dictionary[k_]=sim
        tracks.append(track_genre)
    print 'Done'   
    return Sim_dictionary,tracks

def convertLDA(genres_):
    ldalist={}
    for k,v in genres_.items():  
        #corpus_=corpora.MmCorpus(k_+'__corpus.mm')  
        hdp_ = models.HdpModel.load(k+'_unigram.hdp')
        lda_ = hdp_.hdp_to_lda()
        ldalist[k]=lda_        
    saveDict(ldalist,'lda_list')
    return ldalist       
        
        
def tfidfGenre(genres_):
    tfidfList={}
    for k,v in genres_.items():    
        #dict_ = corpora.Dictionary.load(k_+'__.dict')
        corpus_=corpora.MmCorpus(k+'__corpus.mm')
        tfidf_=tfidf_albums(corpus_)
        tfidfList[k]=tfidf_
    saveDict(tfidfList,'tfidf_list')
    return tfidfList
    
def All_Wrapper(genre_,title,artist_,k):
    listModels=[]
    #listSimDict={}
    #tracksList={}
    tags=[]
    tracks=[]
    track_=input_Song(title,artist_)
    getTags(track_,tags,artist_)
    Sim_dictionary={}
    for k_,v in genre_.items():
        dict_ = corpora.Dictionary.load(k_+'__.dict')
        corpus_=corpora.MmCorpus(k_+'__corpus.mm')
        hdp = models.HdpModel.load(k_+'_unigram.hdp')
        #lda_ = hdp.hdp_to_lda()
        tfidf_ = tfidf_albums(corpus_)
        listModels=[hdp,tfidf_]
        #tokenize tags
        tags1=[]
        for tag in tags:
            tags1+=tokenizeTags(tag,dict_.itervalues())
        #query
        model_Index=0
        for model in listModels:
            print 'model',model_Index
            vec1 = ApplyHDP_tag(tags,dict_,model)
            if model_Index==1:
                sim,track_genre=search_space_tfidf(vec1,v,dict_,corpus_,model,k)
                Sim_dictionary[k_+str(model_Index)]=sim
                tracks.append(track_genre)
            else:
                sim,track_genre=search_space(vec1,v,dict_,corpus_,model,k)
                Sim_dictionary[k_+str(model_Index)]=sim
                tracks.append(track_genre)
            model_Index+=1
        
    print 'Done'   
    return Sim_dictionary,tracks
        
def search_space(vec,albumList,dict_,corpus_,hdp_,k):
    numFeatures=len(hdp_.show_topics(topics=-1))
    index = similarities.SparseMatrixSimilarity(hdp_[corpus_],num_features=numFeatures)
    sim = index[vec]  
    sim = sorted(enumerate(sim),key=lambda item: -item[1])[:k]
    #get top k tracks    
    track = getTopTracks(vec,dict_,hdp_,sim,albumList)
    return sim,track

def search_space_tfidf(vec,albumList,dict_,corpus_,tfidf_,k):
    index = similarities.SparseMatrixSimilarity(tfidf_[corpus_],num_features=len(dict_))
    sim = index[vec]
    sim = sorted(enumerate(sim),key=lambda item: -item[1])[:k]
    #get top k tracks    
    track = getTopTracks(vec,dict_,tfidf_,sim,albumList)
    return sim,track   

def getTopTracks(vec,dict_,hdp,sim,albumList):
    for t in dict(sim):
        album = albumList[t]
        try:
            track_tags = getTracks(album.name,album.artist)
            max_cosine=0
            track_max=''
            for key,val in track_tags.items():
                vec_track=ApplyHDP_tag(val,dict_,hdp)
                cos = CosineVec(vec_track,vec)
                if cos>max_cosine:
                    max_cosine=cos
                    track_max = key
            print track_max
            return track_max
        except Exception:
            pass        


def CosineVec(v1,v2):
    v1=dict(v1)
    v2=dict(v2)
    cosine=0
    for i,j in (v1).items():
        if (v2).__contains__(i):
            cosine+=j*v2.get(i)
    #normalize
    v1_size=0
    v2_size=0
    for t,p in v1.items():
        v1_size += p*p
    for t,p in v2.items():
        v2_size += p*p
    
    return cosine/(math.pow(v1_size,0.5)*math.pow(v2_size,0.5))
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.cluster.bicluster import SpectralCoclustering
#from sklearn.cluster import MiniBatchKMeans


#vectorizer = TfidfVectorizer(min_df=1,tokenizer=tokenize)
#cocluster = SpectralCoclustering(n_clusters=3,svd_method='arpack', random_state=0)
#kmeans = MiniBatchKMeans(n_clusters=3,batch_size=20000,random_state=0)
#X=vectorizer.fit_transform(texts)

#cocluster.fit(X)
#dictionary = corpora.Dictionary(uni_token)
#model = ldaGenerate(uni_token,'tmp')


#test corpusGenerate - by a list of unigrams, a list of unigrams+bigrams
#LDA
