[![License](http://img.shields.io/:license-Apache_v2-blue.svg)](https://github.com/maropu/scavenger/blob/master/LICENSE)

This is an experimental prototype to provide data profiling & cleaning functinalites for Spark catalog tables.

## How to try this plugin

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

    # Runs jobs to repair the seven NULL cells in the `adult` table
    >>> df = scavenger.repair().setTableName("adult").setRowId("tid").run()
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

For more running examples, please check scripts in the [resources/examples](./resources/examples) folder.

## Constraint-based Error Detection

You can use constraint rules to detect error cells if you set a file having the rules in `.setConstraints`.
Note that the file format follows the [HoloClean](http://www.holoclean.io/) one;
they use the [denial constraints](https://dl.acm.org/doi/10.14778/2536258.2536262) whose
predicates cannot hold true simultaneously.

    # Constraints below mean that `Sex="Female"` and `Relationship="Husband"`
    # (`Sex="Male"` and `Relationship="Wife"`) does not hold true simultaneously.
    $ cat ./testdata/adult_constraints.txt
    t1&EQ(t1.Sex,"Female")&EQ(t1.Relationship,"Husband")
    t1&EQ(t1.Sex,"Male")&EQ(t1.Relationship,"Wife")

    # Use the constraints to detect errors then repair them.
    # Note that the query will return repair candidates instead of clean data
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

