# Loads a target data then defines tables for it
boston_schema = "tid string, CRIM double, ZN string, INDUS string, CHAS string, " \
    "NOX string, RM double, AGE string, DIS double, RAD string, TAX string, " \
    "PTRATIO string, B double, LSTAT double"
spark.read \
    .option("header", True) \
    .schema(boston_schema) \
    .csv("./testdata/boston.csv") \
    .write \
    .saveAsTable("boston")

scavenger.misc \
    .options({"db_name": "default", "table_name": "boston", "row_id": "tid"}) \
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
repaired_df = scavenger.repair \
    .setDbName("default") \
    .setTableName("boston") \
    .setRowId("tid") \
    .setDiscreteThreshold(30) \
    .option("hp.no_progress_loss", "300") \
    .run()

# Computes performance numbers for discrete attributes (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(spark.table("boston_clean"), ["tid", "attribute"], "inner")

# Compares predicted values with the correct ones
pdf.orderBy("attribute").show()

is_discrete = "attribute NOT IN ('CRIM', 'LSTAT')"
discrete_pdf = pdf.where(is_discrete)
ground_truth_discrete_df = spark.table("error_cells_ground_truth").where(is_discrete)
discrete_rdf = discrete_pdf.join(ground_truth_discrete_df, ["tid", "attribute"], "right_outer")

precision = discrete_pdf.where("repaired <=> correct_val").count() / discrete_pdf.count()
recall = discrete_pdf.where("repaired <=> correct_val").count() / discrete_pdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print(f"Precision={precision} Recall={recall} F1={f1}")

# Computes performance numbers for continous attributes (RMSE)
is_continous = f"NOT({is_discrete})"
continous_pdf = pdf.where(is_continous)

# Show a scatter plog for repaired/correct_val values
import matplotlib.pyplot as plt
g = continous_pdf.selectExpr("double(repaired)", "double(correct_val)").toPandas().plot.scatter(x="correct_val", y="repaired")
plt.show(g)

n = continous_pdf.count()
rmse = continous_pdf.selectExpr(f"sqrt(sum(pow(correct_val - repaired, 2.0)) / {n}) rmse") \
    .collect()[0] \
    .rmse
mae = continous_pdf.selectExpr(f"sum(abs(correct_val - repaired)) / {n} mae") \
    .collect()[0] \
    .mae

print(f"RMSE={rmse} MAE={mae} RMSE/MAE={rmse/mae}")
