/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.api.python

import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.python._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types._

import io.github.maropu.Utils._

/** A Python API entry point for data cleaning. */
object ScavengerRepairApi extends Logging {

  private case class Metadata(spark: SparkSession) {
    private val kvs = mutable.ArrayBuffer[(String, Any)]()

    def add(key: String, value: Any): Unit = {
      kvs += key -> value
    }

    def toJson: String = {
      kvs.map {
        case (k, v: String) => s""""$k":"$v""""
        case (k, ar: Seq[String]) => s""""$k":${ar.map(v => s""""$v"""").mkString("[", ",", "]")}"""
      }.mkString("{", ",", "}")
    }

    override def toString: String = {
      kvs.map {
        case (k, v: String) => s"""$k=>"$v""""
        case (k, ar: Seq[String]) => s"$k=>${ar.map(v => s""""$v"""").mkString(",")}"
      }.mkString(", ")
    }
  }

  private def logBasedOnLevel(msg: => String): Unit = {
    // This method should be called inside `withSparkSession`
    val spark = SparkSession.getActiveSession.get
    spark.sessionState.conf.logLevel match {
      case "TRACE" => logTrace(msg)
      case "DEBUG" => logDebug(msg)
      case "INFO" => logInfo(msg)
      case "WARN" => logWarning(msg)
      case "ERROR" => logError(msg)
      case _ => logTrace(msg)
    }
  }

  private def createAndCacheTempView(df: DataFrame, name: String = ""): String = {
    def timer(name: String)(computeRowCnt: => Long): Unit = {
      val t0 = System.nanoTime()
      val rowCnt = computeRowCnt
      val t1 = System.nanoTime()
      logBasedOnLevel(s"Elapsed time to compute '$name' with $rowCnt rows: " +
        ((t1 - t0 + 0.0) / 1000000000.0) + "s")
    }
    val tempViewId = getRandomString()
    val numShufflePartitions = df.sparkSession.sessionState.conf.numShufflePartitions
    df.coalesce(numShufflePartitions).cache.createOrReplaceTempView(tempViewId)
    timer(if (name.nonEmpty) s"$name($tempViewId)" else tempViewId) {
      df.sparkSession.table(tempViewId).count()
    }
    tempViewId
  }

  def filterDiscreteAttrs(dbName: String, tableName: String, rowId: String, discreteThres: Int, approxCntEnabled: Boolean): String = {
    logBasedOnLevel(s"filterDiscreteAttrs called with: dbName=$dbName tableName=$tableName rowId=$rowId " +
      s"discreteThres=$discreteThres approxCntEnabled=$approxCntEnabled")

     withSparkSession { sparkSession =>
      // Checks if the given table has a column named `rowId`
      val inputDf = sparkSession.table(if (dbName.nonEmpty) s"$dbName.$tableName" else tableName)
      val tableAttrs = inputDf.schema.map(_.name)
      if (!tableAttrs.contains(rowId)) {
        // TODO: Implicitly adds unique row IDs if they don't exist in a given table
        throw new SparkException(s"Column '$rowId' does not exist in table '$dbName.$tableName'.")
      }

      val queryToComputeStats = {
        val tableStats = {
          val tableNode = inputDf.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
          tableNode.computeStats()
        }
        val attrStatMap = tableStats.attributeStats.map {
          kv => (kv._1.name, kv._2.distinctCount)
        }

        // If we already have stats in a catalog, we just use them
        val rowCount = tableStats.rowCount.map { cnt => s"bigint(${cnt.toLong}) /* rowCount */" }
          .getOrElse("COUNT(1)")
        val distinctCounts = tableAttrs.map { attrName =>
          val aggFunc = if (attrName != rowId && approxCntEnabled) {
            s"APPROX_COUNT_DISTINCT($attrName)"
          } else {
            s"COUNT(DISTINCT $attrName)"
          }
          attrStatMap.get(attrName).map {
            distinctCntOpt => distinctCntOpt.map { v => s"bigint(${v.toLong}) /* $attrName */" }
              .getOrElse(aggFunc)
          }.getOrElse(aggFunc)
        }

        s"""
           |SELECT
           |  $rowCount,
           |  ${distinctCounts.mkString(",\n  ")}
           |FROM
           |  $dbName.$tableName
         """.stripMargin
      }

      logBasedOnLevel(s"Query to compute $dbName.$tableName stats:" + queryToComputeStats)

      val statRow = sparkSession.sql(queryToComputeStats).take(1).head
      val attrsWithStats = tableAttrs.zipWithIndex.map { case (f, i) => (f, statRow.getLong(i + 1)) }
      val rowCnt = statRow.getLong(0)

      logBasedOnLevel({
        val distinctCnts = attrsWithStats.map { case (a, c) => s"distinctCnt($a):$c" }
        s"rowCnt:$rowCnt ${distinctCnts.mkString(" ")}"
      })

      if (attrsWithStats.collectFirst { case (a, cnt) if a == rowId => cnt }.get != rowCnt) {
        throw new SparkException(s"Uniqueness does not hold in column '$rowId' of table '$dbName.$tableName'.")
      }

      def isDiscrete(v: Long) = 1 < v && v < discreteThres
      val (discreteAttrs, nonDiscreteAttrs) = attrsWithStats.sortBy(_._2).partition { a => isDiscrete(a._2) }
      if (nonDiscreteAttrs.size > 1) {
        val droppedCols = nonDiscreteAttrs.filterNot(_._1 == rowId).map { case (a, c) => s"$a($c)" }
        logWarning("Dropped the columns having non-suitable domain size: " +
          droppedCols.mkString(", "))
      }

      // TODO: We could change this value to 64 in v3.1 (SPARK-30279) and we need to
      // support the more number of attributes for repair.
      val maxAttrNumToRepair = 32
      val attrSet = (if (discreteAttrs.size >= maxAttrNumToRepair) {
        val droppedCols = discreteAttrs.drop(maxAttrNumToRepair).map { case (a, c) => s"$a($c)" }
        logWarning(s"Maximum number of attributes is $maxAttrNumToRepair but " +
          s"the ${discreteAttrs.size} discrete attributes found in table '$dbName.$tableName', so " +
          s"the ${droppedCols.size} attributes dropped: ${droppedCols.mkString(", ")}")
        discreteAttrs.take(maxAttrNumToRepair)
      } else {
        discreteAttrs
      }).map(_._1).toSet

      val discreteDf = inputDf.selectExpr(inputDf.columns.filter(attrSet.contains) :+ rowId: _*)
      createAndCacheTempView(discreteDf, "discrete_attrs")
    }
  }

  private def loadConstraintsFromFile(constraintFilePath: String, tableName: String, tableAttrs: Seq[String]): DenialConstraints = {
    // Loads all the denial constraints from a given file path
    val allConstraints = DenialConstraints.parse(constraintFilePath)
    // Checks if all the attributes contained in `constraintFilePath` exist in `table`
    val attrsInConstraints = allConstraints.attrNames
    val tableAttrSet = tableAttrs.toSet
    val absentAttrs = attrsInConstraints.filterNot(tableAttrSet.contains)
    if (absentAttrs.nonEmpty) {
      logBasedOnLevel(s"Non-existent constraint attributes found in $tableName: " +
        absentAttrs.mkString(", "))
      val newPredEntries = allConstraints.entries.filter { _.forall { p =>
        tableAttrSet.contains(p.leftAttr) && tableAttrSet.contains(p.rightAttr)
      }}
      if (newPredEntries.isEmpty) {
        throw new SparkException(s"No valid constraint found in $tableName")
      }
      allConstraints.copy(entries = newPredEntries)
    } else {
      allConstraints
    }
  }

  def detectErrorCells(constraintFilePath: String, dbName: String, tableName: String, rowId: String): String = {
    logBasedOnLevel(s"detectErrorCells called with: constraintFilePath=$constraintFilePath " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      // Checks if the given table has a column named `rowId`
      val inputDf = sparkSession.table(if (dbName.nonEmpty) s"$dbName.$tableName" else tableName)
      val tableAttrs = inputDf.schema.map(_.name)
      if (!tableAttrs.contains(rowId)) {
        // TODO: Implicitly adds unique row IDs if they don't exist in a given table
        throw new SparkException(s"Column '$rowId' does not exist in table '$dbName.$tableName'.")
      }

      withTempView(sparkSession, inputDf, cache = true) { inputView =>
        val tableAttrs = sparkSession.table(inputView).schema.map(_.name)
        val tableAttrNum = sparkSession.table(inputView).schema.length
        val constraints = loadConstraintsFromFile(constraintFilePath, tableName, tableAttrs)
        logBasedOnLevel({
          val constraintLists = constraints.entries.zipWithIndex.map { case (preds, i) =>
            preds.map(_.toString("t1", "t2")).mkString(s" [$i] ", ",", "")
          }
          s"""
             |Loads constraints from '$constraintFilePath':
             |${constraintLists.mkString("\n")}
           """.stripMargin
        })

        // Detects error erroneous cells in a given table
        val tableAttrToId = tableAttrs.zipWithIndex.toMap
        val errCellDf = constraints.entries.flatMap { preds =>
          val queryToValidateConstraint =
            s"""
               |SELECT t1.$rowId `_tid_`
               |FROM $inputView AS t1
               |WHERE EXISTS (
               |  SELECT t2.$rowId
               |  FROM $inputView AS t2
               |  WHERE ${DenialConstraints.toWhereCondition(preds, "t1", "t2")}
               |)
             """.stripMargin

          val df = sparkSession.sql(queryToValidateConstraint)
          logBasedOnLevel(
            s"""
               |Number of violate tuples: ${df.count}
               |Query to validate constraints:
               |$queryToValidateConstraint
             """.stripMargin)

          preds.flatMap { p => p.leftAttr :: p.rightAttr :: Nil }.map { attr =>
            val attrId = tableAttrToId(attr)
            df.selectExpr("_tid_",
              s""""$attr" AS attrName""",
              s"int(_tid_) * int($tableAttrNum) + int($attrId) AS cellId",
              s"int($attrId) AS attr_idx")
          }
        }.reduce(_.union(_)).distinct()

        val errCellView = createAndCacheTempView(errCellDf, "err_cells")
        outputConsole({
          val tableRowCnt = sparkSession.table(inputView).count()
          val errCellNum = sparkSession.table(errCellView).count()
          val totalCellNum = tableRowCnt * tableAttrNum
          val errRatio = (errCellNum + 0.0) / totalCellNum
          s"Found $errCellNum/$totalCellNum error cells (${errRatio * 100.0}%) in attributes " +
            s"(${constraints.attrNames.mkString(", ")}) of $tableName(${tableAttrs.mkString(", ")})"
        })
        errCellView
      }
    }
  }

  def computeAttrStats(
      discreteAttrView: String,
      errCellView: String,
      rowId: String,
      sampleRatio: Double,
      statThresRatio: Double): String = {

    logBasedOnLevel(s"computeAttrStats called with: discreteAttrView=$discreteAttrView " +
      s"errCellView=$errCellView rowId=$rowId sampleRatio=$sampleRatio statThresRatio=$statThresRatio")

    withSparkSession { sparkSession =>
      // Computes numbers for single and pair-wise statistics in the input table
      val tableAttrs = sparkSession.table(discreteAttrView).schema.map(_.name)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        tableAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      val statDf = {
        val groupSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
        val inputDf = if (sampleRatio < 1.0) {
          sparkSession.table(discreteAttrView).sample(sampleRatio)
        } else {
          sparkSession.table(discreteAttrView)
        }
        withTempView(sparkSession, inputDf) { inputView =>
          val filterClauseOption = if (statThresRatio > 0.0) {
            val cond = s"HAVING cnt > ${(inputDf.count * statThresRatio).toInt}"
            logBasedOnLevel(s"Attributes stats filter enabled: $cond")
            cond
          } else {
            ""
          }
          sparkSession.sql(
            s"""
               |SELECT ${tableAttrs.mkString(", ")}, COUNT(1) cnt
               |FROM $inputView
               |GROUP BY GROUPING SETS (
               |  ${tableAttrs.map(a => s"($a)").mkString(", ")},
               |  ${groupSets.map(_.toSeq).map { case Seq(a1, a2) => s"($a1,$a2)" }.mkString(", ")}
               |)
               |$filterClauseOption
             """.stripMargin)
        }
      }

      createAndCacheTempView(statDf, "attr_stats")
    }
  }

  def computePrerequisiteMetadata(
      discreteAttrView: String,
      attrStatView: String,
      errCellView: String,
      rowId: String,
      minCorrThres: Double,
      minAttrsToComputeDomains: Int,
      maxAttrsToComputeDomains: Int,
      defaultMaxDomainSize: Int): String = {

    logBasedOnLevel(s"computeMetadata called with: discreteAttrView=$discreteAttrView " +
      s"attrStatView=$attrStatView errCellView=$errCellView rowId=$rowId minCorrThres=$minCorrThres " +
      s"minAttrsToComputeDomains=$minAttrsToComputeDomains maxAttrsToComputeDomains=$maxAttrsToComputeDomains " +
      s"defaultMaxDomainSize=$defaultMaxDomainSize")

    withSparkSession { sparkSession =>
      val metadata = Metadata(sparkSession)
      val discreteAttrs = sparkSession.table(discreteAttrView).schema.map(_.name).filter(_ != rowId)
      val tableRowCnt = sparkSession.table(discreteAttrView).count()

      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      def whereCaluseToFilterStat(a: String): String =
        s"$a IS NOT NULL AND ${discreteAttrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"

      val pairWiseStats = attrPairsToRepair.map { case attrPair @ (attrToRepair, a) =>
        val pairWiseStatDf = sparkSession.sql(
          s"""
             |SELECT
             |  v1.X, v1.Y, (cntX / $tableRowCnt) pX, (cntY / $tableRowCnt) pY, (cntXY / $tableRowCnt) pXY
             |FROM (
             |  SELECT $attrToRepair X, $a Y, cnt cntXY
             |  FROM $attrStatView
             |  WHERE $attrToRepair IS NOT NULL AND
             |    $a IS NOT NULL
             |) v1, (
             |  SELECT $attrToRepair X, cnt cntX
             |  FROM $attrStatView
             |  WHERE ${whereCaluseToFilterStat(attrToRepair)}
             |) v2, (
             |  /* TODO: Needs to reconsider how-to-handle NULL */
             |  /* Use `MAX` to drop ($a, null) tuples in `$discreteAttrView` */
             |  SELECT $a Y, MAX(cnt) cntY
             |  FROM $attrStatView
             |  WHERE ${whereCaluseToFilterStat(a)}
             |  GROUP BY $a
             |) v3
             |WHERE
             |  v1.X = v2.X AND
             |  v1.Y = v3.Y
           """.stripMargin)

        attrPair -> pairWiseStatDf.selectExpr("SUM(pXY * log2(pXY / (pX * pY)))")
          .collect.map { row =>
            if (!row.isNullAt(0)) row.getDouble(0) else 0.0
          }.head
      }

      val pairWiseStatMap = pairWiseStats.groupBy { case ((attrToRepair, _), _) =>
        attrToRepair
      }.map { case (k, v) =>
        k -> v.map { case ((_, attr), v) =>
          (attr, v)
        }.sortBy(_._2).reverse
      }
      logBasedOnLevel({
        val pairStats = pairWiseStatMap.map { case (k, v) =>
          s"$k(${v.head._2},${v.last._2})=>${v.map(a => s"${a._1}:${a._2}").mkString(",")}"
        }
        s"""
           |Pair-wise statistics:
           |${pairStats.mkString("\n")}
         """.stripMargin
      })

      val corrAttrs = pairWiseStatMap.map { case (k, v) =>
        val attrs = v.filter(_._2 > minCorrThres)
        (k, if (attrs.size > maxAttrsToComputeDomains) {
          attrs.take(maxAttrsToComputeDomains)
        } else if (attrs.size < minAttrsToComputeDomains) {
          // If correlated attributes not found, we pick up data from its domain randomly
          logBasedOnLevel(s"Correlated attributes not found for $k")
          Nil
        } else {
          attrs
        })
      }

      sparkSession.udf.register("extractField", (row: Row, offset: Int) => row.getString(offset))
      val cellExprs = discreteAttrs.map { a => s"CAST(l.$a AS STRING) $a" }
      // Needs to keep the correlated attributes for selecting their domains
      val corrAttrSet = corrAttrs.flatMap(_._2.map(_._1)).toSet
      val corrCols = if (corrAttrSet.nonEmpty) {
        corrAttrSet.mkString(", ", ", ", "")
      } else {
        ""
      }
      val rvDf = sparkSession.sql(
        s"""
           |SELECT
           |  l.$rowId,
           |  cellId,
           |  attr_idx,
           |  attrName,
           |  extractField(struct(${cellExprs.mkString(", ")}), attr_idx) initValue
           |  $corrCols
           |FROM
           |  $discreteAttrView l, $errCellView r
           |WHERE
           |  l.$rowId = r._tid_
         """.stripMargin)

      // TODO: More efficient way to assign unique IDs
      val rvWithIdDf = {
        val rvRdd = rvDf.rdd.zipWithIndex().map { case (r, i) => Row.fromSeq(i +: r.toSeq) }
        val rvSchemaWithId = StructType(StructField("_eid", LongType) +: rvDf.schema)
        sparkSession.createDataFrame(rvRdd, rvSchemaWithId)
      }

      val cellDomainDf = withTempView(sparkSession, rvWithIdDf) { rvView =>
        corrAttrs.map { case (attrName, corrAttrsWithScores) =>
          // Computes domains for error cells
          def computeDomain(stmtToComputeDomain: String) = {
            sparkSession.sql(
              s"""
                 |SELECT
                 |  _eid vid,
                 |  $rowId,
                 |  cellId,
                 |  attr_idx,
                 |  attrName,
                 |  domain,
                 |  size(domain) domainSize,
                 |  0 fixed,
                 |  initValue,
                 |  (array_position(domain, initValue) - int(1)) AS initIndex,
                 |  initValue weakLabel,
                 |  (array_position(domain, initValue) - int(1)) AS weakLabelIndex
                 |FROM (
                 |  SELECT
                 |    rv._eid,
                 |    rv.$rowId,
                 |    rv.cellId,
                 |    rv.attr_idx,
                 |    rv.attrName,
                 |    array_sort(array_union(array(rv.initValue), d.domain)) domain,
                 |    IF(ISNULL(rv.initValue), shuffle(d.domain)[0], rv.initValue) initValue
                 |  FROM (
                 |    SELECT * FROM $rvView WHERE attrName = "$attrName"
                 |  ) rv
                 |  $stmtToComputeDomain
                 |)
               """.stripMargin)
          }

          if (corrAttrsWithScores.nonEmpty) {
            val corrAttrs = corrAttrsWithScores.map(_._1)
            logBasedOnLevel(s"Computing '$attrName' domain from correlated attributes (${corrAttrs.mkString(",")})...")
            val dfs = corrAttrs.zipWithIndex.map { case (attr, i) =>
              sparkSession.sql(
                s"""
                   |/* NOTE: This aggregate query holds a key having NULL */
                   |SELECT $attr, collect_set($attrName) dom$i
                   |FROM $discreteAttrView
                   |GROUP BY $attr
                 """.stripMargin)
            }

            val domainDf = {
              val joinedDf = dfs.tail.foldLeft(dfs.head) { case (a, b) => a.join(b) }
              withTempView(sparkSession, joinedDf) { domainSpaceView =>
                sparkSession.sql(
                  s"""
                     |SELECT
                     |  ${corrAttrs.mkString(", ")},
                     |  concat(${corrAttrs.indices.map(i => s"dom$i").mkString(", ")}) domain
                     |FROM
                     |  $domainSpaceView
                   """.stripMargin)
              }
            }

            withTempView(sparkSession, domainDf) { domainView =>
              computeDomain(
                s"""
                   |LEFT OUTER JOIN
                   |  $domainView d
                   |ON
                   |  /* NOTE: Keys of `$domainView` can have NULL */
                   |  ${corrAttrs.map(v => s"rv.$v <=> d.$v").mkString(" AND ")}
                 """.stripMargin)
            }
          } else {
            // Since any correlated attribute not found, we need to select domains randomly
            val defaultDomainDf = sparkSession.sql(
              s"""
                 |SELECT collect_set($attrName) domain
                 |FROM (
                 |  SELECT $attrName, COUNT(1) cnt
                 |  FROM $discreteAttrView
                 |  GROUP BY $attrName
                 |  ORDER BY cnt DESC
                 |  LIMIT $defaultMaxDomainSize
                 |)
               """.stripMargin)

            withTempView(sparkSession, defaultDomainDf) { defaultDomainView =>
              computeDomain(s", $defaultDomainView d")
            }
          }
        }
      }.reduce(_.union(_))

      val cellDomainView = createAndCacheTempView(cellDomainDf, "cell_domain")
      metadata.add("cell_domain", cellDomainView)

      // Number of rows in `cellDomainView` is the same with the number of error cells
      assert(cellDomainDf.count == sparkSession.table(errCellView).count)

      val weakLabelDf = sparkSession.sql(
        s"""
           |SELECT vid, weakLabel, weakLabelIndex, fixed, /* (t2.cellId IS NULL) */ IF(domainSize > 1, false, true) AS clean
           |FROM $cellDomainView AS t1
           |LEFT OUTER JOIN $errCellView AS t2
           |ON t1.cellId = t2.cellId
           |WHERE weakLabel IS NOT NULL AND (
           |  t2.cellId IS NULL OR t1.fixed != 1
           |)
         """.stripMargin)

      metadata.add("weak_labels", createAndCacheTempView(weakLabelDf, "weak_labels"))

      val varMaskDf = sparkSession.sql(s"SELECT vid, domainSize FROM $cellDomainView")
      metadata.add("var_masks", createAndCacheTempView(varMaskDf, "var_masks"))

      val (totalVars, classes) = sparkSession.sql(s"SELECT COUNT(vid), MAX(domainSize) FROM $cellDomainView")
        .collect.headOption.map { case Row(l: Long, i: Int) => (l, i) }.get

      logBasedOnLevel(s"totalVars=$totalVars classes=$classes " +
        s"featureAttrs(${discreteAttrs.size})=${discreteAttrs.mkString(",")}")
      metadata.add("total_vars", s"$totalVars")
      metadata.add("total_attrs", s"${discreteAttrs.size}")
      metadata.add("feature_attrs", discreteAttrs)
      metadata.add("classes", s"$classes")

      val posValDf = sparkSession.sql(
        s"""
           |SELECT vid, $rowId, cellId, attrName, posexplode(domain) (valId, rVal)
           |FROM $cellDomainView
         """.stripMargin)

      metadata.add("pos_values", createAndCacheTempView(posValDf, "pos_values"))
      metadata.toJson
    }
  }

  // Creates a PyTorch feature:
  //   tensor = -1 * torch.ones(1, classes, total_attrs)
  //   tensor[0][init_idx][attr_idx] = 1.0
  def createInitAttrFeatureView(cellDomainView: String): String = {
    logBasedOnLevel(s"createInitAttrFeatureView called with: cellDomainView=$cellDomainView")

    withSparkSession { sparkSession =>
      val initAttrFtDf = sparkSession.sql(s"SELECT initIndex init_idx, attr_idx FROM $cellDomainView")
      createAndCacheTempView(initAttrFtDf, "init_attr_feature")
    }
  }

  // Creates a PyTorch feature:
  //   torch.zeros(1, classes, total_attrs) = prob
  def createFreqFeatureView(discreteAttrView: String, cellDomainView: String, errCellView: String): String = {
    logBasedOnLevel(s"createFreqFeatureView called with: discreteAttrView=$discreteAttrView " +
      s"cellDomainVIew=$cellDomainView errCellView=$errCellView")

    withSparkSession { sparkSession =>
      val tableRowCnt = sparkSession.table(discreteAttrView).count()
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val freqFtDf = attrsToRepair.map { attr =>
        sparkSession.sql(
          s"""
             |SELECT vid, valId idx, attr_idx, (freq / $tableRowCnt) prob
             |FROM (
             |  SELECT vid, attr_idx, posexplode(domain) (valId, rVal)
             |  FROM $cellDomainView
             |) d, (
             |  SELECT $attr, COUNT(1) freq
             |  FROM $discreteAttrView
             |  GROUP BY $attr
             |) f
             |WHERE
             |  d.rVal = f.$attr
           """.stripMargin)
      }.reduce(_.union(_))

      createAndCacheTempView(freqFtDf, "freq_feature")
    }
  }

  private def whereCaluseToFilterStat(a: String, attrs: Seq[String]): String = {
    s"$a IS NOT NULL AND ${attrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"
  }

  // Creates a PyTorch feature:
  //   torch.zeros(1, classes, total_attrs * total_attrs) = prob
  def createOccurAttrFeatureView(
      discreteAttrView: String,
      errCellView: String,
      cellDomainView: String,
      statView: String,
      rowId: String): String = {

    logBasedOnLevel(s"createOccurAttrFeatureView called with: discreteAttrView=$discreteAttrView " +
      s"cellDomainVIew=$cellDomainView errCellView=$errCellView statView=$statView rowId=$rowId")

    withSparkSession { sparkSession =>
      val tableAttrs = sparkSession.table(discreteAttrView).schema.map(_.name)
      val tableAttrNum = sparkSession.table(discreteAttrView).schema.length
      val tableRowCnt = sparkSession.table(discreteAttrView).count()
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val tableAttrToId = tableAttrs.zipWithIndex.toMap
      val occFtDf = attrsToRepair.indices.flatMap { i =>
        val (Seq((rvAttr, _)), attrs) = attrsToRepair.zipWithIndex.partition { case (_, j) => i == j }
        attrs.map { case (attr, _) =>
          val index = tableAttrToId(rvAttr) * tableAttrNum + tableAttrToId(attr)
          val smoothingParam = 0.001
          sparkSession.sql(
            s"""
               |SELECT
               |  vid,
               |  valId rv_domain_idx,
               |  $index idx,
               |  (cntYX / $tableRowCnt) pYX,
               |  (cntX / $tableRowCnt) pX,
               |  COALESCE(prob, DOUBLE($smoothingParam)) prob
               |FROM (
               |  SELECT vid, valId, rVal, $attr
               |  FROM
               |    $discreteAttrView t, (
               |      SELECT vid, $rowId, posexplode(domain) (valId, rVal)
               |      FROM $cellDomainView
               |      WHERE attrName = '$rvAttr'
               |    ) d
               |  WHERE
               |    t.$rowId = d.$rowId
               |) t1 LEFT OUTER JOIN (
               |  SELECT YX.$rvAttr, X.$attr, cntYX, cntX, (cntYX / cntX) prob
               |  FROM (
               |    SELECT $rvAttr, $attr X, cnt cntYX
               |    FROM $statView
               |    WHERE $rvAttr IS NOT NULL AND
               |      $attr IS NOT NULL
               |  ) YX, (
               |    /* Use `MAX` to drop ($attr, null) tuples in `$discreteAttrView` */
               |    SELECT $attr, MAX(cnt) cntX
               |    FROM $statView
               |    WHERE ${whereCaluseToFilterStat(attr, tableAttrs)}
               |    GROUP BY $attr
               |  ) X
               |  WHERE YX.X = X.$attr
               |) t2
               |ON
               |  t1.rVal = t2.$rvAttr AND
               |  t1.$attr = t2.$attr
             """.stripMargin)
        }
      }.reduce(_.union(_)).orderBy("vid")

      createAndCacheTempView(occFtDf, "occ_attr_feature")
    }
  }

  // Creates a PyTorch feature:
  //   torch.zeros(total_vars, classes, 1) = #violations
  def createConstraintFeatureView(
      constraintFilePath: String,
      discreteAttrView: String,
      errCellView: String,
      posValueView: String,
      rowId: String,
      sampleRatio: Double): String = {

    logBasedOnLevel(s"createConstraintFeatureView called with: constraintFilePath=$constraintFilePath " +
      s"discreteAttrView=$discreteAttrView posValueView=$posValueView rowId=$rowId sampleRatio=$sampleRatio")

    withSparkSession { sparkSession =>
      val tableAttrs = sparkSession.table(discreteAttrView).schema.map(_.name)
      val constraints = loadConstraintsFromFile(constraintFilePath, discreteAttrView, tableAttrs)

      // Filters rows having error cells
      val errRowDf = {
        val inputDf = sparkSession.sql(
          s"""
             |SELECT t1.$rowId, ${constraints.attrNames.map(v => s"t1.$v").mkString(", ")}
             |FROM $discreteAttrView t1, (
             |  SELECT DISTINCT `_tid_` AS $rowId
             |  FROM $errCellView
             |) t2
             |WHERE t1.$rowId = t2.$rowId
             |/*
             |// This query throws an analysis exception in v2.4.x
             |SELECT $rowId, ${constraints.attrNames.mkString(", ")}
             |FROM $discreteAttrView
             |WHERE $rowId IN (
             |  SELECT DISTINCT `_tid_`
             |  FROM $errCellView
             |)
             | */
         """.stripMargin)

        if (sampleRatio < 1.0) {
          inputDf.sample(sampleRatio)
        } else {
          inputDf
        }
      }

      withTempView(sparkSession, errRowDf, cache = true) { errRowView =>
        val metadata = Metadata(sparkSession)
        val predicates = mutable.ArrayBuffer[(String, String)]()
        val offsets = constraints.entries.scanLeft(0) { case (idx, preds) => idx + preds.size }.init
        val queries = constraints.entries.zip(offsets).flatMap { case (preds, offset) =>
          preds.indices.map { i =>
            val (Seq((violationPred, _)), fixedPreds) = preds.zipWithIndex.partition { case (_, j) => i == j }
            val fixedWhereCaluses = DenialConstraints.toWhereCondition(fixedPreds.map(_._1), "t1", "t2")
            predicates += ((fixedWhereCaluses, violationPred.toString("t1", "t2")))
            val rvAttr = violationPred.rightAttr
            val queryToCountViolations =
              s"""
                 |SELECT
                 |  ${offset + i} constraintId, vid, valId, COUNT(1) violations
                 |FROM
                 |  $errRowView AS t1, $errRowView AS t2, $posValueView AS t3
                 |  /* $discreteAttrView AS t1, $discreteAttrView AS t2, $posValueView AS t3 */
                 |WHERE
                 |  t1.$rowId != t2.$rowId AND
                 |  t1.$rowId = t3.$rowId AND
                 |  t3.attrName = '$rvAttr' AND
                 |  $fixedWhereCaluses AND
                 |  t3.rVal = t2.$rvAttr
                 |GROUP BY
                 |  vid, valId
               """.stripMargin

            logBasedOnLevel(queryToCountViolations)
            queryToCountViolations
          }
        }

        val constraintFtDf = queries.zipWithIndex.map { case (query, i) =>
          outputConsole(s"Starts processing the $i/${queries.size} query to compute #violations...")
          sparkSession.sql(query)
        }.reduce(_.union(_))

        metadata.add("constraint_feature", createAndCacheTempView(constraintFtDf, "constraint_feature"))
        metadata.add("fixed_preds", predicates.map(_._1))
        metadata.add("violation_preds", predicates.map(_._2))
        metadata.toJson
      }
    }
  }
}
