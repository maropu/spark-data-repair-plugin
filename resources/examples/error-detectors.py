# Loads input tables for the examples of the built-in error detectors
spark.read \
    .option("header", True) \
    .csv("./testdata/adult.csv") \
    .write \
    .saveAsTable("adult")

spark.read \
    .option("header", True) \
    .csv("./testdata/hospital.csv") \
    .write \
    .saveAsTable("hospital")

boston_schema = "tid int, CRIM double, ZN int, INDUS string, CHAS string, " \
    "NOX string, RM double, AGE string, DIS double, RAD string, TAX int, " \
    "PTRATIO string, B double, LSTAT double"
spark.read \
    .option("header", True) \
    .schema(boston_schema) \
    .csv("./testdata/boston.csv") \
    .write \
    .saveAsTable("boston")

# Imports all the built-in error detectors
from repair.errors import NullErrorDetector

# For `NullErrorDetector`
error_cells_df = delphi.repair \
    .setTableName("hospital") \
    .setRowId("tid") \
    .setErrorDetectors([NullErrorDetector()]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `DomainValues`
error_cells_df = delphi.repair \
    .setTableName("adult") \
    .setRowId("tid") \
    .setErrorDetectors([DomainValues(attr='Sex', values=['Male', 'Female'])]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# A 'autofill' mode - we assume domain values tend to appear frequently against illegal values
target_columns = ['MeasureCode', 'ZipCode', 'City']

domain_value_error_detectors = []
for c in target_columns:
    domain_value_error_detectors.append(DomainValues(attr=c, autofill=True, min_count_thres=12))

error_cells_df = delphi.repair \
    .setTableName("hospital") \
    .setRowId("tid") \
    .setErrorDetectors(domain_value_error_detectors) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `RegExErrorDetector`
error_cells_df = delphi.repair \
    .setTableName("hospital") \
    .setRowId("tid") \
    .setErrorDetectors([RegExErrorDetector(attr='ZipCode', regex='\\d\\d\\d\\d\\d')]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `ConstraintErrorDetector`
target_columns = ['MeasureName', 'ZipCode', 'EmergencyService', 'CountyName']

error_cells_df = delphi.repair \
    .setTableName("hospital") \
    .setRowId("tid") \
    .setTargets(target_columns) \
    .setErrorDetectors([ConstraintErrorDetector(constraint_path="./testdata/hospital_constraints.txt")]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `GaussianOutlierErrorDetector`
error_cells_df = delphi.repair \
    .setTableName("boston") \
    .setRowId("tid") \
    .setErrorDetectors([GaussianOutlierErrorDetector(approx_enabled=False)]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `LOFOutlierErrorDetector`
error_cells_df = delphi.repair \
    .setTableName("boston") \
    .setRowId("tid") \
    .setErrorDetectors([LOFOutlierErrorDetector()]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)

# For `ScikitLearnBackedErrorDetector`
from sklearn.neighbors import LocalOutlierFactor
error_cells_df = delphi.repair \
    .setTableName("boston") \
    .setRowId("tid") \
    .setErrorDetectors([ScikitLearnBackedErrorDetector(lambda: LocalOutlierFactor(novelty=False))]) \
    .run(detect_errors_only=True)

error_cells_df.show(3)
