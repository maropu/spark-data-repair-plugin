# Loads a target data then defines tables for it
iris_schema = "tid string, sepal_length double, sepal_width double, petal_length double, petal_width double"
spark.read \
  .option("header", True) \
  .schema(iris_schema) \
  .csv("./testdata/iris.csv") \
  .write \
  .saveAsTable("iris")

scavenger.repair().misc() \
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

spark.table("iris_flatten") \
  .join(spark.table("iris_clean"), ["tid", "attribute"], "inner") \
  .where("not(value <=> correct_val)") \
  .write \
  .saveAsTable("error_cells_ground_truth")

spark.table("iris_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
repaired_df = scavenger.repair() \
  .setDbName("default") \
  .setTableName("iris") \
  .setRowId("tid") \
  .setInferenceOrder("entropy") \
  .run(return_repair_candidates=True)

# Computes performance numbers for continous attributes (RMSE)
n = repaired_df.count()
rmse = repaired_df \
  .join(spark.table("iris_clean"), ["tid", "attribute"], "inner") \
  .selectExpr("sqrt(sum(pow(correct_val - repaired, 2.0)) / %s) rmse" % n) \
  .collect()[0] \
  .rmse

print("RMSE=%s" % rmse)

