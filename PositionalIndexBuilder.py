
import collections
import math
import os 
import nltk 
from nltk.stem import PorterStemmer 
from natsort import natsorted 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize  
from collections import defaultdict
from nltk.tokenize import RegexpTokenizer


class PositionalIndexBuilder:
    def __init__(self , folderName) :
        self.folderName = folderName;
        nltk.download('punkt')
        nltk.download('stopwords')
        self.stopWords = set(stopwords.words('english'))  
        self.stemmer = PorterStemmer()
        self.tokenizer = RegexpTokenizer(r'\w+') 
        self.corpus = [];
        self.corpusAfterTokenization = [];
        self.corpusNormalizedAndTokenized = []
        self.positionalIndex = defaultdict(list);

    def readFile(self,fileName) :
        notebook_dirctory=os.getcwd()
        file=open(notebook_dirctory+"/" + self.folderName + "/" + fileName,'r')
        dataFromFile=file.read()
        file.close()     
        self.corpus.append(dataFromFile) 
    
    def tokenizeAndNormalize(self) :
        for i in range(0 , len(self.corpus)) :
            self.corpus[i] = self.tokenizer.tokenize(self.corpus[i]) ;
            for x in range(len(self.corpus[i])) :
                self.corpus[i][x] = self.corpus[i][x].lower();
        
        for i in self.corpus :
            temp1 = [];
            temp2 = [];
            
            for j in i :
                
                if j not in self.stopWords :

                    temp1.append(j) ;
                
                temp2.append(j)
            
            self.corpusNormalizedAndTokenized.append((temp1))
            
            self.corpusAfterTokenization.append(temp2)


    def buildPositionalIndex(self):
        notebook_dirctory=os.getcwd()
        fileNo = 0
        file_map = {} 

        files = natsorted(os.listdir(notebook_dirctory + "\\"+ self.folderName)) ;
        for i in files :
            self.readFile(i);
        self.tokenizeAndNormalize()

        for corpus in self.corpusNormalizedAndTokenized :

            for index , word in enumerate(corpus) :
                stemmedTerm = self.stemmer.stem(word);
                if stemmedTerm in self.positionalIndex: 
                        
                        # Increment total freq by 1. 
                        self.positionalIndex[stemmedTerm][0] = self.positionalIndex[stemmedTerm][0] + 1
                        
                        # Check if the term has existed in that DocID before. 
                        if fileNo in self.positionalIndex[stemmedTerm][1]: 
                            self.positionalIndex[stemmedTerm][1][fileNo].append(index) 
                            
                        else: 
                            self.positionalIndex[stemmedTerm][1][fileNo] = [index] 

                    # If term does not exist in the positional index dictionary  
                    # (first encounter). 
                else: 
                        
                    # Initialize the list. 
                    self.positionalIndex[stemmedTerm] = [] 
                    # The total frequency is 1. 
                    self.positionalIndex[stemmedTerm].append(1) 
                    # The postings list is initially empty. 
                    self.positionalIndex[stemmedTerm].append({})       
                    # Add doc ID to postings list. 
                    self.positionalIndex[stemmedTerm][1][fileNo] = [index] 
            file_map[fileNo] = notebook_dirctory+ "/" + self.folderName + "/" + i 

            fileNo+=1   

    
    def queryPhrase(self,word_one,word_two):
        word_one = self.stemmer.stem(word_one)
        word_two = self.stemmer.stem(word_two)

        dict_1 = {};
        dict_2 = {};
        try :

            dict_1=self.positionalIndex[word_one][1]
        except Exception:
            pass
        try :

            dict_2=self.positionalIndex[word_two][1]
        except Exception :
            pass
        
        doc_nums=list(set(dict_1.keys()).intersection(set(dict_2.keys())))
        final=[]
        for i in doc_nums:
            index_1=dict_1[i]
            index_2=dict_2[i]
            
            for x in index_1:
                for z in index_2:
                # print(x,'  ',z)
                    if(x+1) == z:
                        final.append(("doc "+str(i+1),x,x+1))
        if len(final)== 0 :
            return "query doesn't exist"
        return final




