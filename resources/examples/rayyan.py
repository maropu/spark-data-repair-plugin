# Loads a target data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/rayyan.csv") \
    .write \
    .saveAsTable("rayyan")

delphi.misc \
    .options({"db_name": "default", "table_name": "rayyan", "row_id": "id"}) \
    .flatten() \
    .write \
    .saveAsTable("rayyan_flatten")

spark.table("rayyan").show(1)
spark.table("rayyan_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/rayyan_clean.csv") \
    .write \
    .saveAsTable("rayyan_clean")

spark.table("rayyan_flatten") \
    .join(spark.table("rayyan_clean"), ["id", "attribute"], "inner") \
    .where("not(value <=> correct_val)") \
    .write \
    .saveAsTable("error_cells_ground_truth")

spark.table("rayyan_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
repaired_df = delphi.repair \
    .setDbName("default") \
    .setTableName("rayyan") \
    .setRowId("id") \
    .setErrorCells("error_cells_ground_truth") \
    .setDiscreteThreshold(400) \
    .option("hp.no_progress_loss", "100") \
    .run()

# Computes performance numbers (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(spark.table("rayyan_clean"), ["id", "attribute"], "inner")
rdf = repaired_df.join(
    spark.table("error_cells_ground_truth"),
    ["id", "attribute"], "right_outer")

# Compares predicted values with the correct ones
pdf.orderBy("attribute").show()

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall + 0.0001)

print("Precision={} Recall={} F1={}".format(precision, recall, f1))
