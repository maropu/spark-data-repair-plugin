# Loads a target data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/tax.csv") \
    .write \
    .saveAsTable("tax")

delphi.misc \
    .options({"db_name": "default", "table_name": "tax", "row_id": "tid"}) \
    .flatten() \
    .write \
    .saveAsTable("tax_flatten")

spark.table("tax").show(1)
spark.table("tax_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/tax_clean.csv") \
    .write \
    .saveAsTable("tax_clean")

spark.table("tax_flatten") \
    .join(spark.table("tax_clean"), ["tid", "attribute"], "inner") \
    .where("not(value <=> correct_val)") \
    .write \
    .saveAsTable("error_cells_ground_truth")

spark.table("tax_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Shows column stats
delphi.misc.options({"table_name": "tax"}).describe().show()

# Detects error cells then repairs them
repaired_df = delphi.repair \
    .setDbName("default") \
    .setTableName("tax") \
    .setRowId("tid") \
    .setErrorCells("error_cells_ground_truth") \
    .setTargets(["state", "marital_status", "has_child"]) \
    .setDiscreteThreshold(300) \
    .option("model.hp.no_progress_loss", "100") \
    .run()

# Computes performance numbers (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(spark.table("tax_clean"), ["tid", "attribute"], "inner")
rdf = repaired_df.join(
    spark.table("error_cells_ground_truth").where("attribute IN ('state', 'marital_status', 'has_child')"),
    ["tid", "attribute"], "right_outer")

# Compares predicted values with the correct ones
pdf.orderBy("attribute").show()

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print("Precision={} Recall={} F1={}".format(precision, recall, f1))
