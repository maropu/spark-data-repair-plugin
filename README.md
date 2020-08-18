[![License](http://img.shields.io/:license-Apache_v2-blue.svg)](https://github.com/maropu/scavenger/blob/master/LICENSE)
[![Build and test](https://github.com/maropu/scavenger/workflows/Build%20and%20tests/badge.svg)](https://github.com/maropu/scavenger/actions?query=workflow%3A%22Build+and+tests%22)
<!---
[![Coverage Status](https://coveralls.io/repos/github/maropu/scavenger/badge.svg?branch=master)](https://coveralls.io/github/maropu/scavenger?branch=master)
-->

This is an experimental prototype to provide data repair functinalites for Spark catalog tables.
Clean and consistent data can have a positive impact on downstream anaysis;
clean data make reporting and machine learning more accurate and
consistent data with constraints (e.g., functional dependency) are important for efficient query plans.
Therefore, data repairing is a first step in an anaysis pipeline and
this plugin intends to implement a scalable repair algorithm on Spark.

## How to Repair Error Cells

```
$ git clone https://github.com/maropu/scavenger.git
$ cd scavenger

# This repository includes a simple wrapper script `bin/python` to create
# a virtual environment to resolve the required dependencies (e.g., Python 3.6 and PySpark 3.0),
# then launch a Python VM with this plugin.
$ ./bin/python

Welcome to
      ____              __
     / __/__  ___ _____/ /__
    _\ \/ _ \/ _ `/ __/  '_/
   /__ / .__/\_,_/_/ /_/\_\   version 3.0.0
      /_/

Using Python version 3.6.8 (default, Dec 29 2018 19:04:46)
SparkSession available as 'spark'.
Scavenger APIs (version 0.1.0-spark3.0-EXPERIMENTAL) available as 'scavenger'.

# Loads CSV data and defines a table in a catalog
>>> spark.read.option("header", True).csv("./testdata/adult.csv").write.saveAsTable("adult")
>>> spark.table("adult").show()
+---+-----+------------+-----------------+-------------+------+-------------+-----------+
|tid|  Age|   Education|       Occupation| Relationship|   Sex|      Country|     Income|
+---+-----+------------+-----------------+-------------+------+-------------+-----------+
|  0|31-50|Some-college|     Craft-repair|      Husband|  Male|United-States|LessThan50K|
|  1|  >50|Some-college|  Exec-managerial|    Own-child|Female|United-States|LessThan50K|
|  2|31-50|   Bachelors|            Sales|      Husband|  Male|United-States|LessThan50K|
|  3|22-30|     HS-grad|     Craft-repair|    Own-child|  null|United-States|LessThan50K|
|  4|22-30|     HS-grad|  Farming-fishing|      Husband|Female|United-States|LessThan50K|
|  5| null|Some-college|     Craft-repair|      Husband|  Male|United-States|       null|
|  6|31-50|     HS-grad|   Prof-specialty|Not-in-family|Female|United-States|LessThan50K|
|  7|31-50| Prof-school|   Prof-specialty|      Husband|  null|        India|MoreThan50K|
|  8|18-21|Some-college|     Adm-clerical|    Own-child|Female|United-States|LessThan50K|
|  9|  >50|     HS-grad|  Farming-fishing|      Husband|  Male|United-States|LessThan50K|
| 10|  >50|   Assoc-voc|   Prof-specialty|      Husband|  Male|United-States|LessThan50K|
| 11|  >50|     HS-grad|            Sales|      Husband|Female|United-States|MoreThan50K|
| 12| null|   Bachelors|  Exec-managerial|      Husband|  null|United-States|MoreThan50K|
| 13|22-30|     HS-grad|     Craft-repair|Not-in-family|  Male|United-States|LessThan50K|
| 14|31-50|  Assoc-acdm|  Exec-managerial|    Unmarried|  Male|United-States|LessThan50K|
| 15|22-30|Some-college|            Sales|    Own-child|  Male|United-States|LessThan50K|
| 16|  >50|Some-college|  Exec-managerial|    Unmarried|Female|United-States|       null|
| 17|31-50|     HS-grad|     Adm-clerical|Not-in-family|Female|United-States|LessThan50K|
| 18|31-50|        10th|Handlers-cleaners|      Husband|  Male|United-States|LessThan50K|
| 19|31-50|     HS-grad|            Sales|      Husband|  Male|         Iran|MoreThan50K|
+---+-----+------------+-----------------+-------------+------+-------------+-----------+

# Runs jobs to compute repair updates for the seven NULL cells in the `adult` table.
# A 'repaired' column represents proposed updates to repiar them.
>>> repair_updates_df = scavenger.repair().setTableName("adult").setRowId("tid").run()
>>> repair_updates_df.show()
+---+---------+-------------+-----------+
|tid|attribute|current_value|   repaired|
+---+---------+-------------+-----------+
|  7|      Sex|         null|     Female|
| 12|      Age|         null|      18-21|
| 12|      Sex|         null|     Female|
|  3|      Sex|         null|     Female|
|  5|      Age|         null|      18-21|
|  5|   Income|         null|MoreThan50K|
| 16|   Income|         null|MoreThan50K|
+---+---------+-------------+-----------+

# You need to set `True` to `repair_data` for getting repaired data
>>> df = scavenger.repair().setTableName("adult").setRowId("tid").run(repair_data=True)
>>> df.show()
+---+-----+------------+-----------------+-------------+------+-------------+-----------+
|tid|  Age|   Education|       Occupation| Relationship|   Sex|      Country|     Income|
+---+-----+------------+-----------------+-------------+------+-------------+-----------+
|  0|31-50|Some-college|     Craft-repair|      Husband|  Male|United-States|LessThan50K|
|  1|  >50|Some-college|  Exec-managerial|    Own-child|Female|United-States|LessThan50K|
|  2|31-50|   Bachelors|            Sales|      Husband|  Male|United-States|LessThan50K|
|  3|22-30|     HS-grad|     Craft-repair|    Own-child|  Male|United-States|LessThan50K|
|  4|22-30|     HS-grad|  Farming-fishing|      Husband|Female|United-States|LessThan50K|
|  5|31-50|Some-college|     Craft-repair|      Husband|  Male|United-States|LessThan50K|
|  6|31-50|     HS-grad|   Prof-specialty|Not-in-family|Female|United-States|LessThan50K|
|  7|31-50| Prof-school|   Prof-specialty|      Husband|  Male|        India|MoreThan50K|
|  8|18-21|Some-college|     Adm-clerical|    Own-child|Female|United-States|LessThan50K|
|  9|  >50|     HS-grad|  Farming-fishing|      Husband|  Male|United-States|LessThan50K|
| 10|  >50|   Assoc-voc|   Prof-specialty|      Husband|  Male|United-States|LessThan50K|
| 11|  >50|     HS-grad|            Sales|      Husband|Female|United-States|MoreThan50K|
| 12|31-50|   Bachelors|  Exec-managerial|      Husband|  Male|United-States|MoreThan50K|
| 13|22-30|     HS-grad|     Craft-repair|Not-in-family|  Male|United-States|LessThan50K|
| 14|31-50|  Assoc-acdm|  Exec-managerial|    Unmarried|  Male|United-States|LessThan50K|
| 15|22-30|Some-college|            Sales|    Own-child|  Male|United-States|LessThan50K|
| 16|  >50|Some-college|  Exec-managerial|    Unmarried|Female|United-States|LessThan50K|
| 17|31-50|     HS-grad|     Adm-clerical|Not-in-family|Female|United-States|LessThan50K|
| 18|31-50|        10th|Handlers-cleaners|      Husband|  Male|United-States|LessThan50K|
| 19|31-50|     HS-grad|            Sales|      Husband|  Male|         Iran|MoreThan50K|
+---+-----+------------+-----------------+-------------+------+-------------+-----------+

# Or, you can use a misc function to apply computed repair updates into the input
>>> df = scavenger.repair().setTableName("adult").setRowId("tid") \
...   .setRepairUpdates(repair_updates_df) \
...   .run()

>>> df.show()
<the same output above>
```

For more running examples, please check scripts in the [resources/examples](./resources/examples) folder.

## Error Detection

You can use constraint rules to detect error cells if you set a file having the rules in `.setConstraints`.
Note that the file format follows the [HoloClean](http://www.holoclean.io/) one;
they use the [denial constraints](https://www.sciencedirect.com/science/article/pii/S0890540105000179) [5]
whose predicates cannot hold true simultaneously.

```
# Constraints below mean that `Sex="Female"` and `Relationship="Husband"`
# (`Sex="Male"` and `Relationship="Wife"`) does not hold true simultaneously.
$ cat ./testdata/adult_constraints.txt
t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")
t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")

# Use the constraints to detect errors then repair them.
# Note that a process will return repair candidates instead of clean data
# when setting `True` to `return_repair_candidates`.
>>> df = scavenger.repair().setTableName("adult").setRowId("tid") \
...   .setConstraints("./testdata/adult_constraints.txt") \
...   .run(return_repair_candidates=True)

# Changes values from `Female` to `Male` in the `Sex` cells
# of the 4th and 11th rows.
>>> df.show()
+---+------------+-------------+-----------+
|tid|   attribute|current_value|   repaired|
+---+------------+-------------+-----------+
|  3|         Sex|         null|       Male|
|  4|Relationship|      Husband|    Husband|
|  4|         Sex|       Female|       Male|
|  5|         Age|         null|      31-50|
|  5|      Income|         null|LessThan50K|
|  7|         Sex|         null|       Male|
| 11|Relationship|      Husband|    Husband|
| 11|         Sex|       Female|       Male|
| 12|         Age|         null|      31-50|
| 12|         Sex|         null|       Male|
| 16|      Income|         null|LessThan50K|
+---+------------+-------------+-----------+
```

If you want to know detected error cells, you can set `True` to `detect_errors_only`
for getting them in pre-processing as follows;

```
# Runs jobs to detect error cells
>>> df = scavenger.repair().setTableName("adult").setRowId("tid").run(detect_errors_only=True)
>>> df.show()
+---+---------+-------------+
|tid|attribute|current_value|
+---+---------+-------------+
| 12|      Age|         null|
|  5|      Age|         null|
| 12|      Sex|         null|
|  7|      Sex|         null|
|  3|      Sex|         null|
| 16|   Income|         null|
|  5|   Income|         null|
+---+---------+-------------+
```

## Configurations

```
scavenger.repair()

  // Basic Parameters
  .setDbName(str)                              // database name (default: 'default')
  .setTableName(str)                           // table name
  .setRowId(str)                               // unique column name in table
  .setConstraints(str)                         // path of constraint file

  // Parameters for Error Detection
  .setErrorCells(df)                           // user-specified error cells
  .setDiscreteThreshold(float)                 // max domain size of discrete values (default: 80)
  .setMinCorrThreshold(float)                  // threshold to decide which columns are used to compute domains (default: 0.70)
  .setDomainThresholds(float, float)           // thresholds to reduce domain size (default: 0.0, 0.70)
  .setAttrMaxNumToComputeDomains(int)          // max number of attributes to compute posterior probabiity based on the Naive Bayes assumption (default: 4)
  .setAttrStatSampleRatio(float )              // sample ratio for table used to compute co-occurrence frequency (default: 1.0)
  .setAttrStatThreshold(float)                 // threshold for filtering out low frequency (default: 0.0)

  // Parameters for Repair Model Training
  .setTrainingDataSampleRatio(float)           // sample ratio for table used to build statistical models (default: 1.0)
  .setMaxTrainingColumnNum(int)                // max number of columns used to build models
  .setSmallDomainThreshold(int)                // max domain size for low-cardinality catogory encoding (default: 12)
  .setInferenceOrder(str)                      // how to order target columns to build models (default: 'entropy')

  // Parameters for Repairing
  .setRepairUpdates(df)                        // user-specified repair updates
  .setMaximalLikelihoodRepairEnabled(boolean)  // whether to enable maximal likelihood repair (default: False)
  .setRepairDelta(int)                         // max number of applied repairs

  // Running Mode Parameters
  .run(
    detect_errors_only=boolean,                // whether to return detected error cells (default: False)
    compute_repair_candidate_prob=boolean      // whether to return probabiity mass function of repairs (default: False)
    repair_data=boolean,                       // whether to return repaired data (default: False)
  )
```

## References

See [REFERENCES](./REFERENCES.md).

## Bug reports

If you hit some bugs and requests, please leave some comments on [Issues](https://github.com/maropu/scavenger/issues)
or Twitter([@maropu](http://twitter.com/#!/maropu)).

