# charity_news_stance_classifier

謝安的碩士論文實驗程式碼

## 目錄

+ Root
  + Stance_cls_attention.ipynb: 帶注意力機制的LSTM媒體立場分類器
  + Stance_cls_bilstm_bert.ipynb: 使用了Bert文件向量的BiLSTM分類器(論文中不使用)
  + Stance_cls_cnn.ipynb: CNN媒體立場分類器
  + Stance_cls_cnn_bert.ipynb: 使用了Bert文件向量的CNN媒體立場分類器(論文中不使用)
  + Stance_cls_lstm.ipynb: LSTM媒體立場分類器(沒有開啟雙向)
  + Stance_cls_svm.ipynb: SVM媒體立場分類器
  + Stance_cls_svm_bert.ipynb: 使用了Bert文件向量的SVM媒體立場分類器(論文中不使用)
  + thesis_models.py: 以上全部分類器的Pytorch Model都存在這裡。
   
+ Preprocessing: 文件前處理使用的程式碼。
  + preprocessing_keyword_analysis: 用來做關鍵字主題測試，並計算三種關鍵字提取的Accuracy, precision, recall, f1
  + preprocessing_keyword_analysis_plot: 用來給關鍵字抽取測試結果繪圖用的
  + preprocessing_keyword_extract: 使用LDA提取關鍵字
  + preprocessing_tagger: 原始檔案前處理、斷詞等等。
   
+ Results: 實驗結果的data table、圖片，以及表現較好的Model權重檔等等。
   
+ vec_data: 文件轉換向量後存出來的npz檔，方便實驗使用。
   
+ word2vec: 文件轉換向量用的程式碼，標籤編碼也放在這裡。
  + bertvec: 使用bert製作文件向量
  + label_encoding: 標籤向量化
  + vec_processing: 將關鍵字向量分別以相加法/平均法製作成文件向量。
  + word2vec: 將關鍵字轉換成關鍵字向量。
