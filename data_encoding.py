import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

df_filtered = pd.read_csv('clean_data.csv', sep='\t')

df_filtered_dict = df_filtered.to_dict(orient='records')

#print(df_filtered_dict[0])

dv_X = DictVectorizer(sparse=False)

X_encoded = dv_X.fit_transform(df_filtered_dict)

#print(X_encoded[0])

vocab = dv_X.vocabulary_
#print(vocab["WWID"])
#print(X_encoded[0][vocab["WWID"]])

sorted_X = [x[0] for x in sorted(vocab.items(), key=lambda kv: kv[1])]
#print(sorted_X[0])

#print(X_encoded[vocab["Termination_Reason"]])
#print(len(vocab))

df = pd.DataFrame(data=X_encoded, columns=sorted_X)

#print(df.info())
#print(df.columns)
print(df.WWID)

#dftrain, dfeval = np.array_split(df, 2)
dftrain = df

y_train = dftrain.pop('Termination_Reason')

print(y_train)

#y_eval = dfeval.pop('Termination_Reason')

#model = LogisticRegression()
#model.fit(dftrain, y_train)