# Loads a target data then defines tables for it
iris_schema = "tid STRING, sepal_length DOUBLE, sepal_width DOUBLE, " \
    "petal_length DOUBLE, petal_width DOUBLE"
spark.read \
    .option("header", True) \
    .schema(iris_schema) \
    .csv("./testdata/iris.csv") \
    .write \
    .saveAsTable("iris")

scavenger.misc \
    .setDbName("default") \
    .setTableName("iris") \
    .setRowId("tid") \
    .flatten() \
    .write \
    .saveAsTable("iris_flatten")

spark.table("iris").show(1)
spark.table("iris_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/iris_clean.csv") \
    .write \
    .saveAsTable("iris_clean")

spark.table("iris_clean").show(1)

# Detects error cells then repairs them
repaired_df = scavenger.repair \
    .setDbName("default") \
    .setTableName("iris") \
    .setRowId("tid") \
    .setInferenceOrder("entropy") \
    .run()

# Computes performance numbers for continous attributes (RMSE)
n = repaired_df.count()
rmse = repaired_df \
    .join(spark.table("iris_clean"), ["tid", "attribute"], "inner") \
    .selectExpr(f"sqrt(sum(pow(correct_val - repaired, 2.0)) / {n}) rmse") \
    .collect()[0] \
    .rmse

print(f"RMSE={rmse}")
