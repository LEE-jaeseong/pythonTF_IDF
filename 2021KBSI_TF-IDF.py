import pandas as pd # 데이터프레임 사용을 위해
import numpy as np
from math import log # IDF 계산을 위해
import os
#https://wikidocs.net/31698

os.getcwd()
os.chdir('C://Users/owner/Documents/research/KBSI 연구참여')


ntis = pd.read_csv('kbsi과제.csv',error_bad_lines=False, engine='python', encoding='CP949') #한글 깨짐 현상 해결
ntis.dtypes #변수명 확인

projName = ntis['과제명']

ntis.iloc[:,4:8]
docs = ntis.iloc[:,4]+' '+ntis.iloc[:,5]+' '+ntis.iloc[:,6]+' '+ntis.iloc[:,7] # 텍스트 파일 합치기

docLists = np.array(docs.tolist()) # 리스트로 변환

# 단어 분리
vocab = list(set(w for doc in docLists for w in doc.split()))
vocab.sort()


N = len(docLists) # 총 문서의 수

# 사용자 정의함수 생성
def tf(t, d):
    return d.count(t)

def idf(t):
    df = 0
    for doc in docLists:
        df += t in doc
    return log(N/(df + 1))

def tfidf(t, d):
    return tf(t,d)* idf(t)


# tf 계산
result = []
for i in range(N): # 각 문서에 대해서 아래 명령을 수행
    result.append([])
    d = docLists[i]
    for j in range(len(vocab)):
        t = vocab[j]
        result[-1].append(tf(t, d))

tf_ = pd.DataFrame(result, columns = vocab)
tf_

# idf 계산
result = []
for j in range(len(vocab)):
    t = vocab[j]
    result.append(idf(t))

idf_ = pd.DataFrame(result, index = vocab, columns = ["IDF"])
idf_


# tf-idf 계산
result = []
for i in range(N):
    result.append([])
    d = docLists[i]
    for j in range(len(vocab)):
        t = vocab[j]

        result[-1].append(tfidf(t,d))

tfidf_ = pd.DataFrame(result, columns = vocab)




# 상위 n개 value의 index 추출 https://ddiri01.tistory.com/313

def which(self):
    try:
        self = list(iter(self))
    except TypeError as e:
        raise Exception("""'which' method can only be applied to iterables.
        {}""".format(str(e)))
    indices = [i for i, x in enumerate(self) if bool(x) == True]
    return(indices)



result = []
for k in range(N):
    idx = tfidf_.iloc[k,:]
    threshold = int(len(which(idx)) * 0.2)
    if threshold == 0:
        threshold = len(which(idx))

    topIdx = sorted(range(len(idx)), key=lambda i: idx[i])[-threshold:]
    output = pd.DataFrame(data={'proj':np.repeat(projName.iloc[k],threshold), 'keyword':np.array(vocab)[topIdx],'value':list(tfidf_.iloc[k,topIdx])})
    result.append(output)
    print(k)

finalOutput = pd.concat(result)

finalOutput.to_csv('kbsiProj3.csv', index=False, encoding='cp949') #encoding 옵션: csv 파일에서 한글 (컬럼 혹은 내용) 읽어올 때 encoding='cp949' (혹은 encoding='euc-kr') 옵션 사용, 엑셀의 csv 파일 utf-8 인코딩 (유니코드) 인식 버그
