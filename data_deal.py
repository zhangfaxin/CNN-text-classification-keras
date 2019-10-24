import pandas as pd
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    df = pd.read_csv('./data/train_data.csv')
    df2 = pd.read_csv('./data/test_data.csv')
    df = df.append(df2).drop_duplicates(subset=['review'])
    x_train,x_test,y_train,y_test = train_test_split(df.review,df.label,test_size=0.3)
    train_data = pd.DataFrame({'review':x_train,'label':y_train}).reset_index(drop=True)
    test_data = pd.DataFrame({'review':x_test,'label':y_test}).reset_index(drop=True)
    train_data.to_csv('./train_data_1.csv',header=True,index=False)
    test_data.to_csv('./test_data_1.csv',header=True,index=False)
    print(df.head())
