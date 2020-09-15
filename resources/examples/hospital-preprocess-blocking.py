from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.feature import *

df = spark.table("hospital").limit(100)
df = df.selectExpr("*", "array(%s) AS _input" % ", ".join(df.columns))

ngram = NGram(n=2, inputCol="_input", outputCol="_ngrams")
df = ngram.transform(df).drop("_input")

cv = CountVectorizer(inputCol="_ngrams", outputCol="_features")
model = cv.fit(df)
df = model.transform(df).drop("_ngrams")

pca = PCA(k=3, inputCol="_features", outputCol="_pcaFeatures")
model = pca.fit(df)
df = model.transform(df).drop("_features").withColumnRenamed("_pcaFeatures", "_features")

bkm = BisectingKMeans().setFeaturesCol("_features").setPredictionCol("_k").setK(3).setSeed(0)
model = bkm.fit(df)
df = model.transform(df).drop("_features")

