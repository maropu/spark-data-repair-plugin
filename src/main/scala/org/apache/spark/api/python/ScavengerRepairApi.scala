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

import org.apache.spark.SparkException
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.types._
import org.apache.spark.util.{Utils => SparkUtils}

import io.github.maropu.Utils._

/** A Python API entry point for data cleaning. */
object ScavengerRepairApi extends BaseScavengerRepairApi {

  /**
   * To compare result rows easily, this method flattens an input table
   * as a schema (tid, attribute, val).
   */
  def flattenAsDataFrame(dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"flattenAsDataFrame called with: dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { _ =>
      val (inputDf, _, _) = checkInputTable(dbName, tableName, rowId)
      val expr = inputDf.schema.filter(_.name != rowId)
        .map { f => s"STRUCT($rowId, '${f.name}', CAST(${f.name} AS STRING))" }
        .mkString("ARRAY(", ", ", ")")
      inputDf.selectExpr(s"INLINE($expr) AS (tid, attribute, val)")
    }
  }

  def injectNullAt(dbName: String, tableName: String, targetAttrList: String, nullRatio: Double): DataFrame = {
    logBasedOnLevel(s"flattenAsDataFrame called with: dbName=$dbName tableName=$tableName " +
      s"targetAttrList=$targetAttrList, nullRatio=$nullRatio")

    withSparkSession { _ =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName)
      val targetAttrSet = if (targetAttrList.nonEmpty) {
        val attrSet = SparkUtils.stringToSeq(targetAttrList).toSet
        if (!tableAttrs.exists(attrSet.contains)) {
          throw new SparkException(s"No target attribute selected in $inputName")
        }
        attrSet
      } else {
        tableAttrs.toSet
      }
      val exprs = inputDf.schema.map {
        case f if targetAttrSet.contains(f.name) =>
          s"IF(rand() > $nullRatio, ${f.name}, NULL) AS ${f.name}"
        case f =>
          f.name
      }
      inputDf.selectExpr(exprs: _*)
    }
  }

  def filterDiscreteAttrs(
      dbName: String,
      tableName: String,
      rowId: String,
      blackAttrList: String,
      discreteThres: Int,
      approxCntEnabled: Boolean): String = {

    logBasedOnLevel(s"filterDiscreteAttrs called with: dbName=$dbName tableName=$tableName rowId=$rowId " +
      s"blackAttrList=$blackAttrList discreteThres=$discreteThres approxCntEnabled=$approxCntEnabled")

    withSparkSession { sparkSession =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName, rowId, blackAttrList)
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
      if (discreteAttrs.size < 2) {
        val errMsg = if (discreteAttrs.nonEmpty) {
          s"found: ${discreteAttrs.map(_._1).mkString(",")}"
        } else {
          "no discrete attribute found."
        }
        throw new SparkException(s"$inputName must have more than one discrete attributes, but $errMsg")
      }

      // TODO: We could change this value to 64 in v3.1 (SPARK-30279) and we need to
      // support the more number of attributes for repair.
      val maxAttrNumToRepair = 32
      val attrSet = (if (discreteAttrs.size >= maxAttrNumToRepair) {
        val droppedCols = discreteAttrs.drop(maxAttrNumToRepair).map { case (a, c) => s"$a($c)" }
        logWarning(s"Maximum number of attributes is $maxAttrNumToRepair but " +
          s"the ${discreteAttrs.size} discrete attributes found in table '$dbName.$tableName', so " +
          s"the ${droppedCols.size} attributes dropped: ${droppedCols.mkString(",")}")
        discreteAttrs.take(maxAttrNumToRepair)
      } else {
        discreteAttrs
      }).map(_._1).toSet

      val discreteDf = inputDf.selectExpr(inputDf.columns.filter(attrSet.contains) :+ rowId: _*)
      createAndCacheTempView(discreteDf, "discrete_attrs")
    }
  }

  def createRepairCandidates(discreteAttrView: String, cellDomainView: String, distView: String, rowId: String): String = {
    logBasedOnLevel(s"createRepairCandidates called with: discreteAttrView=$discreteAttrView " +
      s"cellDomainView=$cellDomainView distView=$distView")

    withSparkSession { sparkSession =>
      val repairCandidateDf = sparkSession.sql(
        s"""
           |SELECT
           |  $rowId,
           |  attrName AS attribute,
           |  arrays_zip(domain, dist) dist,
           |  /* TODO: Use array_sort in v3.0 */
           |  domain[array_position(dist, array_max(dist)) - 1] inferred
           |FROM
           |  $distView d
           |INNER JOIN
           |  $cellDomainView c
           |ON
           |  d.vid = c.vid
         """.stripMargin)

      createAndCacheTempView(repairCandidateDf, "repair_candidates")
    }
  }

  def filterCleanRows(dbName: String, tableName: String, rowId: String, errCellView: String): DataFrame = {
    logBasedOnLevel(s"filterCleanRows called with: dbName=$dbName tableName=$tableName " +
      s"rowId=$rowId errCellView=$errCellView")
    withSparkSession { sparkSession =>
      val (inputDf, _, tableAttrs) = checkInputTable(dbName, tableName, rowId)

      withTempView(inputDf) { inputView =>
        sparkSession.sql(
          s"""
             |SELECT t1.$rowId, ${tableAttrs.map(v => s"t1.$v").mkString(", ")}
             |FROM $inputView t1, (
             |  SELECT DISTINCT `_tid_` AS $rowId
             |  FROM $errCellView
             |) t2
             |WHERE t1.$rowId = t2.$rowId
             |/*
             |// This query throws an analysis exception in v2.4.x
             |SELECT t1.$rowId, ${tableAttrs.map(v => s"t1.$v").mkString(", ")}
             |FROM $inputView
             |WHERE $rowId IN (
             |  SELECT DISTINCT `_tid_`
             |  FROM $errCellView
             |)
             | */
         """.stripMargin)
      }
    }
  }

  def repairTableFrom(repairCandidateTable: String, dbName: String, tableName: String, rowId: String): DataFrame = {
    logBasedOnLevel(s"repairTableFrom called with: repairCandidateTable=$repairCandidateTable " +
      s"dbName=$dbName tableName=$tableName rowId=$rowId")

    withSparkSession { sparkSession =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName, rowId)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attribute) FROM $repairCandidateTable")
          .collect.head.getSeq[String](0).toSet
      }

      outputToConsole({
        val repairNum =  sparkSession.table(repairCandidateTable).count()
        val totalCellNum = inputDf.count() * tableAttrs.length
        val repairRatio = (repairNum + 0.0) / totalCellNum
        s"Repairing $repairNum/$totalCellNum error cells (${repairRatio * 100.0}%) of " +
          s"${attrsToRepair.size}/${tableAttrs.size} attributes (${attrsToRepair.mkString(",")}) " +
          s"in the input '$inputName'"
      })

      val repairDf = sparkSession.sql(
        s"""
           |SELECT
           |  $rowId, map_from_entries(COLLECT_LIST(r)) AS repairs
           |FROM (
           |  select $rowId, struct(attribute, inferred) r
           |  FROM $repairCandidateTable
           |)
           |GROUP BY
           |  $rowId
         """.stripMargin)

      withTempView(inputDf) { inputView =>
        withTempView(repairDf) { repairView =>
          val cleanAttrs = tableAttrs.map {
            case attr if attr == rowId =>
              s"$inputView.$rowId"
            case attr if attrsToRepair.contains(attr) =>
              s"IF(ISNOTNULL(repairs['$attr']), repairs['$attr'], $attr) AS $attr"
            case cleanAttr =>
              cleanAttr
          }
          sparkSession.sql(
            s"""
               |SELECT ${cleanAttrs.mkString(", ")}
               |FROM $inputView
               |LEFT OUTER JOIN $repairView
               |ON $inputView.$rowId = $repairView.$rowId
             """.stripMargin)
        }
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
        withTempView(inputDf) { inputView =>
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

      // TODO: Needs to revisit this feature selection
      val featureAttrs = if (true) {
        discreteAttrs.filter { attr =>
          attrsToRepair.contains(attr) || corrAttrSet.contains(attr)
        }
      } else {
        discreteAttrs
      }

      val ftAttrToId = featureAttrs.zipWithIndex.toMap
      val cellDomainDf = withTempView(rvWithIdDf) { rvView =>
        corrAttrs.map { case (attrName, corrAttrsWithScores) =>
          // Computes domains for error cells
          def computeDomain(stmtToComputeDomain: String) = {
            sparkSession.sql(
              s"""
                 |SELECT
                 |  _eid vid,
                 |  $rowId,
                 |  cellId,
                 |  ${ftAttrToId(attrName)} feature_idx,
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
            logBasedOnLevel(s"Computing '$attrName' domain from ${corrAttrs.size} correlated " +
              s"attributes (${corrAttrs.mkString(",")})...")
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
              withTempView(joinedDf) { domainSpaceView =>
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

            withTempView(domainDf) { domainView =>
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

            withTempView(defaultDomainDf) { defaultDomainView =>
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
        s"featureAttrs(${featureAttrs.size})=${featureAttrs.mkString(",")}")
      metadata.add("total_vars", s"$totalVars")
      metadata.add("feature_attrs", featureAttrs)
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
}
