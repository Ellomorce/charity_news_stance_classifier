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
+ Results: 實驗結果的圖片，以及表現較好的Model權重檔等等。
+ vec_data: 文件轉換向量後存出來的npz檔，方便實驗使用。
+ word2vec: 文件轉換向量用的程式碼，標籤編碼也放在這裡。

未完待補
