# Loads a target data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/adult.csv") \
    .write \
    .saveAsTable("adult")

scavenger.misc \
    .options({"db_name": "default", "table_name": "adult", "row_id": "tid"}) \
    .flatten() \
    .write \
    .saveAsTable("adult_flatten")

spark.table("adult").show(1)
spark.table("adult_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/adult_clean.csv") \
    .write \
    .saveAsTable("adult_clean")

spark.table("adult_flatten") \
    .join(spark.table("adult_clean"), ["tid", "attribute"], "inner") \
    .where("not(value <=> correct_val)") \
    .write \
    .saveAsTable("error_cells_ground_truth")

spark.table("adult_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
from repair.detectors import ConstraintErrorDetector
repaired_df = scavenger.repair \
    .setDbName("default") \
    .setTableName("adult") \
    .setRowId("tid") \
    .setErrorDetector(ConstraintErrorDetector(
        constraint_path="./testdata/adult_constraints.txt")) \
    .run()

# Computes performance numbers (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(spark.table("adult_clean"), ["tid", "attribute"], "inner")
rdf = repaired_df.join(spark.table("error_cells_ground_truth"), ["tid", "attribute"], "right_outer")

# Compares predicted values with the correct ones
pdf.show()

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print(f"Precision={precision} Recall={recall} F1={f1}")
