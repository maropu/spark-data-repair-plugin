# Loads a target data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/hospital.csv") \
    .write \
    .saveAsTable("hospital")

delphi.misc \
    .options({"db_name": "default", "table_name": "hospital", "row_id": "tid"}) \
    .flatten() \
    .write \
    .saveAsTable("hospital_flatten")

spark.table("hospital").show(1)
spark.table("hospital_flatten").show(1)

# Loads a ground truth data then defines tables for it
spark.read \
    .option("header", True) \
    .csv("./testdata/hospital_clean.csv") \
    .write \
    .saveAsTable("hospital_clean")

spark.table("hospital_flatten") \
    .join(spark.table("hospital_clean"), ["tid", "attribute"], "inner") \
    .where("not(value <=> correct_val)") \
    .write \
    .saveAsTable("error_cells_ground_truth")

spark.table("hospital_clean").show(1)
spark.table("error_cells_ground_truth").show(1)

# Detects error cells then repairs them
from repair.detectors import NullErrorDetector, ConstraintErrorDetector
error_detectors = [
    ConstraintErrorDetector(constraint_path="./testdata/hospital_constraints.txt"),
    NullErrorDetector()
]
repaired_df = delphi.repair \
    .setDbName("default") \
    .setTableName("hospital") \
    .setRowId("tid") \
    .setErrorDetectors(error_detectors) \
    .setDiscreteThreshold(100) \
    .setRuleBasedModelEnabled(True) \
    .option("hp.no_progress_loss", "100") \
    .run()

# Computes performance numbers (precision & recall)
#  - Precision: the fraction of correct repairs, i.e., repairs that match
#    the ground truth, over the total number of repairs performed
#  - Recall: correct repairs over the total number of errors
pdf = repaired_df.join(
    spark.table("hospital_clean").where("attribute != 'Score'"),
    ["tid", "attribute"], "inner")
rdf = repaired_df.join(
    spark.table("error_cells_ground_truth").where("attribute != 'Score'"),
    ["tid", "attribute"], "right_outer")

# Compares predicted values with the correct ones
pdf.orderBy("attribute").show()

precision = pdf.where("repaired <=> correct_val").count() / pdf.count()
recall = rdf.where("repaired <=> correct_val").count() / rdf.count()
f1 = (2.0 * precision * recall) / (precision + recall)

print("Precision={} Recall={} F1={}".format(precision, recall, f1))
