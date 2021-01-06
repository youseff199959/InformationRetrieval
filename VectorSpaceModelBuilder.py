from collections import defaultdict;
import math
import nltk 
from nltk.stem import PorterStemmer 
from natsort import natsorted 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer

class VectorSpaceModelBuilder :
    def __init__(self ,rawData, data , positionalIndex) :
        self.rawData = rawData
        self.data = data
        self.tf = [];
        self.idf = [];
        self.TFIDF = [];
        self.positionalIndex = positionalIndex

    def findAllTermFreq(self) :
        self.tf = defaultdict(list)
        count = 0;
        for i in self.rawData :
            for x in i :
                self.tf[x] = [0 for f in range(len(self.rawData))]
        for i in range(len(self.rawData) ):
            for x in self.rawData[i] :
                
                count = self.rawData[i].count(x)
                self.tf[x][i] = round(float(count),4);
        return self.tf



    def findIDF(self):
        ## idf of a term = log10(totalNumOfDocs / numberOfDocsWithTerm);
        self.idf = defaultdict(list);
        for i in range(len(self.rawData )):
            
            for x in self.rawData[i] :
                
                numberOfDocsWithTerm = 0
                for f in self.rawData :
                    if x in f :
                        numberOfDocsWithTerm += 1;
                        
                self.idf[x] = round(math.log10(float(len(self.data) / numberOfDocsWithTerm)),4);
        
        return self.idf;

    
    def getTFIDF(self):
        self.TFIDF = defaultdict(float);
        for i in self.rawData :
            for x in i :
                self.TFIDF[x] = [0 for f in range(len(self.rawData))]
        for i in range(len(self.rawData) ):
            for x in self.rawData[i] :
                if len(x) <= 1 :
                    continue
                
                self.TFIDF[x][i] = round(self.tf[x][i] * self.idf[x] , 6);
        
        return self.TFIDF;
    
    def findSimilarity(self , query , doc) :
        similarity = 0;

        vectorQ = [];
        vectorD = [];

        tempQ= word_tokenize(query)
        tempD = word_tokenize(doc);
        sw = set(stopwords.words('english'))  
        tokenizedAndNormalizedQuery = {word for word in tempQ if word not in sw}
        tokenizedAndNormalizedDoc = {word for word in tempD if word not in sw};

        vector = tokenizedAndNormalizedDoc.union(tokenizedAndNormalizedQuery);
        for word in vector :
            if word in tokenizedAndNormalizedDoc:
                vectorQ.append(1);
            else :
                vectorQ.append(0)
            if word in tokenizedAndNormalizedQuery :
                vectorD.append(1);
            else :
                vectorD.append(0)
        

        
        for i in range(len(vector)) :
            similarity += vectorD[i] * vectorQ[i];
        
        answer = similarity / float((sum(vectorD)*sum(vectorQ))**0.5) 
        print(sum(vectorD))
        print(sum(vectorQ))
        return answer

    def findSimilarityBetweenQueryAndAllDocs(self,query) :
        allSimilarity = []
        for i in range(len(self.rawData)):
            allSimilarity.append((i+1 , self.findSimilarity(query , " ".join(self.rawData[i]))));
        
        allSimilarity.sort(key=lambda x : x[-1] , reverse=True)
        return allSimilarity

