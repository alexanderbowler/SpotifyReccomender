#should try to find reccs given certain spotify song

import numpy as np 
import pandas as pd 

def readCSV(filename):
    df = pd.read_csv(filename,header=0,encoding='utf-8')
    return df


def findSong(songName, df):
    '''Finds the song in the dataframe that corresonds
    to the name and artist
    '''
    return df.loc[df['track_name'].str.lower()==songName.lower()]

def prepDFSmall(df):
    newDF = df[['track_name','artist(s)_name','released_year','bpm','key'
                ,'danceability_%','valence_%','energy_%','acousticness_%'
                ,'instrumentalness_%','liveness_%','speechiness_%']]
    #newDF['track_name'] = newDF['track_name'].str.lower()
    return newDF

def prepDFLarge(df):
    newDF = df[['artists','album_name','track_name','popularity','duration_ms',
                'explicit','danceability','energy','key','loudness','mode','speechiness'
                ,'acousticness','instrumentalness','liveness','valence','tempo'
                ,'time_signature','track_genre']]
    #newDF['track_name'] = newDF['track_name'].lower()
    return newDF

def getSimilarSmall(songName,df,weights,artistMatters,yearMatters):
    '''Should find the top 5 most similar songs with certain flags turned on/off'''
    normdf = cleanDFSmall(df)
    orgInfo = findSong(songName,normdf)

    normdf = normdf.drop(normdf.index[normdf['track_name'].str.lower()==songName.lower()])
    if yearMatters:
        normdf = normdf.loc[normdf['released_year']==orgInfo.iloc[0]['released_year']]
    if artistMatters:
        normdf = normdf.loc[normdf['artist(s)_name']==orgInfo.iloc[0]['artist(s)_name']]

    normdf['key'] = normdf['key']==orgInfo.iloc[0]['key'] 
    normdf['key'] = normdf['key'].replace([True,False],[1,0])
    #orgInfo['key'] = 1
    
    normdf = normdf.drop(columns=['artist(s)_name','track_name','released_year'])
    orgInfo = orgInfo.drop(columns=['artist(s)_name','track_name','released_year'])
    orgInfo['key'] = 1
    orgVec = orgInfo.to_numpy()
    npDF = normdf.to_numpy()
    orgVec = orgVec.T
    #print(npDF[0,:].reshape(npDF.shape[1],1))
    #print(orgVec.shape)
    minDifference = compVecs(npDF[0,:].reshape(npDF.shape[1],1),orgVec)
    minIndex = 0
    for i in range(1,npDF.shape[0]):
        if compVecs(npDF[i,:].reshape(npDF.shape[1],1),orgVec) < minDifference:
            minDifference = compVecs(npDF[i,:].reshape(npDF.shape[1],1),orgVec)
            minIndex = i

    print("Index of most similar song ", minIndex, " with a difference of ", minDifference)

def compVecs(vec1:np.array, vec2:np.array):#, weights:np.array):
    if vec1.shape != vec2.shape:
        raise Exception("Vec shapes not equal")
    return np.linalg.norm(vec1-vec2)#*weights


def cleanDFSmall(df):
    df['bpm'] = (df['bpm']-df['bpm'].min())/(df['bpm'].max()-df['bpm'].min())
    df[['danceability_%','valence_%','energy_%','acousticness_%'
                ,'instrumentalness_%','liveness_%','speechiness_%']] = df[['danceability_%','valence_%','energy_%','acousticness_%'
                ,'instrumentalness_%','liveness_%','speechiness_%']]/100
    return df


def runSpotifyRec():
    #spotifyDF = readCSV("spotify-2023.csv")
    spotifyDF = pd.read_csv("spotify-2023.csv",header=0,encoding='utf-8') # for small dataset
    spotifyDF = prepDFSmall(spotifyDF)
        # spotifyDF = pd.read_csv("largeSpotifyDataset.csv",header=0,encoding='utf-8') # for small dataset
    # spotifyDF = prepDFLarge(spotifyDF)
    #sName = input() for input
    sName = 'cruel summer' # for testing
    sName = sName.lower()
    weights = np.ones(len(spotifyDF.columns))
    getSimilarSmall(sName,spotifyDF,weights,True,True)

runSpotifyRec()


