# Loads a target data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/beers.csv") \
    .write \
    .saveAsTable("beers")

delphi.misc \
    .options({"db_name": "default", "table_name": "beers", "row_id": "index"}) \
    .flatten() \
    .write \
    .saveAsTable("beers_flatten")

spark.table("beers").show(1)
spark.table("beers_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/raha/beers_clean.csv") \
    .write \
    .saveAsTable("beers_clean")

spark.table("beers_flatten") \
    .join(spark.table("beers_clean"), ["index", "attribute"], "inner") \
    .where("not(value <=> correct_val)") \
    .write \
    .saveAsTable("error_cells_ground_truth")

spark.table("beers_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
# NOTE: Erroneous attributes are 'ibu', 'abv', 'state', 'ounces', and 'city',
# but the attributes except for 'state' have simple format errors.
repaired_df = delphi.repair \
    .setDbName("default") \
    .setTableName("beers") \
    .setRowId("index") \
    .setErrorCells("error_cells_ground_truth") \
    .setTargets(["state"]) \
    .setDiscreteThreshold(600) \
    .option("hp.no_progress_loss", "100") \
    .run()

# Computes performance numbers (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(spark.table("beers_clean"), ["index", "attribute"], "inner")
rdf = repaired_df.join(
    spark.table("error_cells_ground_truth").where("attribute = 'state'"),
    ["index", "attribute"], "right_outer")

# Compares predicted values with the correct ones
pdf.orderBy("attribute").show()

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print("Precision={} Recall={} F1={}".format(precision, recall, f1))
