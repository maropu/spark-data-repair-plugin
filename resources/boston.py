# Loads a target data then defines tables for it
boston_schema = "tid string, CRIM double, ZN string, INDUS string, CHAS string, NOX string, RM double, AGE string, DIS double, RAD string, TAX string, PTRATIO string, B double, LSTAT double"
spark.read \
  .option("header", True) \
  .schema(boston_schema) \
  .csv("./testdata/boston.csv") \
  .write \
  .saveAsTable("boston")

scavenger.repair().misc() \
  .setDbName("default") \
  .setTableName("boston") \
  .setRowId("tid") \
  .flatten() \
  .write \
  .saveAsTable("boston_flatten")

spark.table("boston").show(1)
spark.table("boston_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
  .option("header", True) \
  .csv("./testdata/boston_clean.csv") \
  .write \
  .saveAsTable("boston_clean")

spark.table("boston_flatten") \
  .join(spark.table("boston_clean"), ["tid", "attribute"], "inner") \
  .where("not(value <=> correct_val)") \
  .write \
  .saveAsTable("error_cells_ground_truth")

spark.table("boston_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
scavenger.repair() \
  .setDbName("default") \
  .setTableName("boston") \
  .setRowId("tid") \
  .setDiscreteThreshold(1000) \
  .run(return_repair_candidates=True) \
  .write \
  .saveAsTable("boston_repaired")

# Computes performance numbers for discrete attributes (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
is_discrete = "attribute NOT IN ('CRIM', 'LSTAT')"
discrete_df = spark.table("boston_repaired").where(is_discrete)
pdf = discrete_df.join(spark.table("boston_clean"), ["tid", "attribute"], "inner")
ground_truth_df = spark.table("error_cells_ground_truth").where(is_discrete)
rdf = discrete_df.join(ground_truth_df, ["tid", "attribute"], "right_outer")

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print("Precision=%s Recall=%s F1=%s" % (precision, recall, f1))

# Computes performance numbers for continous attributes (RMSE)
is_continous = "NOT(%s)" % is_discrete
n = spark.table("boston_repaired").count()
rmse = spark.table("boston_repaired") \
  .where(is_continous) \
  .join(spark.table("boston_clean"), ["tid", "attribute"], "inner") \
  .selectExpr("sqrt(sum(pow(correct_val - repaired, 2.0)) / %s) rmse" % n) \
  .collect()[0] \
  .rmse

print("RMSE=%s" % rmse)

