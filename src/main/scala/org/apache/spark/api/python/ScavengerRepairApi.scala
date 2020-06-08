/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
v* this work for additional information regarding copyright ownership.
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

  private def distinctStatMap(inputName: String): Map[String, Long] = {
    assert(SparkSession.getActiveSession.nonEmpty)
    val spark = SparkSession.getActiveSession.get
    val df = spark.table(inputName)
    val tableStats = {
      val tableNode = df.queryExecution.analyzed.collectLeaves()
        .head.asInstanceOf[LeafNode]
      tableNode.computeStats()
    }
    val statMap = tableStats.attributeStats.map {
      kv => (kv._1.name, kv._2.distinctCount.map(_.toLong).get)
    }
    if (!df.columns.forall(statMap.contains)) {
      throw new SparkException(s"'$inputName' should be analyzed first")
    }
    statMap
  }

  def analyzeAndFilterDiscreteAttrs(
      dbName: String,
      tableName: String,
      rowId: String,
      blackAttrList: String,
      discreteThres: Int): String = {

    logBasedOnLevel(s"filterDiscreteAttrs called with: dbName=$dbName tableName=$tableName rowId=$rowId " +
      s"blackAttrList=$blackAttrList discreteThres=$discreteThres")

    withSparkSession { sparkSession =>
      val (inputDf, inputName, tableAttrs) = checkInputTable(dbName, tableName, rowId, blackAttrList)

      val statMap = {
        // To compute the number of distinct values, runs `ANALYZE TABLE` first
        sparkSession.sql(
          s"""
             |ANALYZE TABLE $inputName COMPUTE STATISTICS
             |FOR COLUMNS ${tableAttrs.mkString(", ")}
           """.stripMargin)

        distinctStatMap(inputName)
      }

      val attrsWithStats = tableAttrs.map { a => (a, statMap(a)) }
      val rowCnt = statMap(rowId)

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

      outputToConsole(s"Loaded $rowCnt rows with ${rowCnt * discreteAttrs.size} cells")

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
           |  d.$rvId = c.$rvId
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
             |  SELECT DISTINCT $rowId
             |  FROM $errCellView
             |) t2
             |WHERE t1.$rowId = t2.$rowId
             |/*
             |// This query throws an analysis exception in v2.4.x
             |SELECT t1.$rowId, ${tableAttrs.map(v => s"t1.$v").mkString(", ")}
             |FROM $inputView
             |WHERE $rowId IN (
             |  SELECT DISTINCT $rowId
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
        val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
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
               |  ${pairSets.map(_.toSeq).map { case Seq(a1, a2) => s"($a1,$a2)" }.mkString(", ")}
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

    logBasedOnLevel(s"computePrerequisiteMetadata called with: discreteAttrView=$discreteAttrView " +
      s"attrStatView=$attrStatView errCellView=$errCellView rowId=$rowId minCorrThres=$minCorrThres " +
      s"minAttrsToComputeDomains=$minAttrsToComputeDomains maxAttrsToComputeDomains=$maxAttrsToComputeDomains " +
      s"defaultMaxDomainSize=$defaultMaxDomainSize")

    withSparkSession { sparkSession =>
      val metadata = Metadata(sparkSession)
      val discreteAttrs = sparkSession.table(discreteAttrView).schema.map(_.name).filter(_ != rowId)
      val rowCnt = sparkSession.table(discreteAttrView).count()

      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      def whereCaluseToFilterStat(a: String): String =
        s"$a IS NOT NULL AND ${discreteAttrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"

      // Computes the conditional entropy: H(x|y) = H(x,y) - H(y).
      // H(x,y) denotes H(x U y). If H(x|y) = 0, then y determines x, i.e., y -> x.
      val hXYs = {
        val pairSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
        pairSets.map { attrPairKey =>
          attrPairKey -> {
            val Seq(a1, a2) = attrPairKey.toSeq
            val df = sparkSession.sql(
              s"""
                 |SELECT -SUM(hXY) hXY
                 |FROM (
                 |  SELECT $a1 X, $a2 Y, (cnt / $rowCnt) * log10(cnt / $rowCnt) hXY
                 |  FROM $attrStatView
                 |  WHERE $a1 IS NOT NULL AND
                 |    $a2 IS NOT NULL
                 |)
             """.stripMargin)

            df.collect().map { row =>
              if (!row.isNullAt(0)) row.getDouble(0) else 0.0
            }.head
          }
        }
      }.toMap

      val hYs = discreteAttrs.map { attrKey =>
        attrKey -> {
          val df = sparkSession.sql(
            s"""
               |SELECT -SUM(hY) hY
               |FROM (
               |  /* TODO: Needs to reconsider how-to-handle NULL */
               |  /* Use `MAX` to drop ($attrKey, null) tuples in `$discreteAttrView` */
               |  SELECT $attrKey Y, (MAX(cnt) / $rowCnt) * log10(MAX(cnt) / $rowCnt) hY
               |  FROM $attrStatView
               |  WHERE ${whereCaluseToFilterStat(attrKey)}
               |  GROUP BY $attrKey
               |)
           """.stripMargin)

          df.collect().map { row =>
            if (!row.isNullAt(0)) row.getDouble(0) else 0.0
          }.head
        }
      }.toMap

      // Uses the domain size of X as a log base for normalization
      val domainStatMap = distinctStatMap(discreteAttrView)

      val pairWiseStats = attrPairsToRepair.map { case attrPair @ (rvX, rvY) =>
        // The conditional entropy is 0 for strongly correlated attributes and 1 for completely independent
        // attributes. We reverse this to reflect the correlation.
        val domainSize = domainStatMap(rvX)
        attrPair -> (1.0 - ((hXYs(Set(rvX, rvY)) - hYs(rvY)) / scala.math.log10(domainSize)))
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

      val attrToId = discreteAttrs.zipWithIndex.toMap
      sparkSession.udf.register("extractField", (row: Row, attrName: String) => {
        row.getString(attrToId(attrName))
      })
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
           |  attrName,
           |  extractField(struct(${cellExprs.mkString(", ")}), attrName) initValue
           |  $corrCols
           |FROM
           |  $discreteAttrView l, $errCellView r
           |WHERE
           |  l.$rowId = r.$rowId
         """.stripMargin)

      // TODO: More efficient way to assign unique IDs
      val rvWithIdDf = {
        val rvRdd = rvDf.rdd.zipWithIndex().map { case (r, i) => Row.fromSeq(i +: r.toSeq) }
        val rvSchemaWithId = StructType(StructField(rvId, LongType) +: rvDf.schema)
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

      val cellDomainDf = withTempView(rvWithIdDf) { rvView =>
        corrAttrs.map { case (attrName, corrAttrsWithScores) =>
          // Adds an empty domain for initial state
          val initDomainDf = sparkSession.sql(
            s"""
               |SELECT $rvId, $rowId, attrName, initValue, array() domain $corrCols
               |FROM $rvView
               |WHERE attrName = '$attrName'
             """.stripMargin)

          val domainSpaceDf = if (corrAttrsWithScores.nonEmpty) {
            val corrAttrs = corrAttrsWithScores.map(_._1)
            logBasedOnLevel(s"Computing '$attrName' domain from ${corrAttrs.size} correlated " +
              s"attributes (${corrAttrs.mkString(",")})...")

            corrAttrs.foldLeft(initDomainDf) { case (df, attr) =>
              withTempView(df) { domainSpaceView =>
                val tau = {
                  // `tau` becomes a threshold on co-occurrence frequency
                  val productSpaceSize = domainStatMap(attr) * domainStatMap(attrName)
                  (0.80 * (rowCnt / productSpaceSize)).toLong
                }
                sparkSession.sql(
                  s"""
                     |SELECT
                     |  $rvId,
                     |  $rowId,
                     |  attrName,
                     |  initValue,
                     |  concat(l.domain, IF(ISNOTNULL(r.d), r.d, array())) domain,
                     |  ${corrAttrSet.map(a => s"l.$a").mkString(",")}
                     |FROM
                     |  $domainSpaceView l
                     |LEFT OUTER JOIN (
                     |  SELECT $attr, collect_set($attrName) d
                     |  FROM (
                     |    SELECT *
                     |    FROM $attrStatView
                     |    WHERE $attrName IS NOT NULL AND
                     |      $attr IS NOT NULL AND
                     |      cnt > $tau
                     |  )
                     |  GROUP BY
                     |    $attr
                     |) r
                     |ON
                     |  l.$attr = r.$attr
                   """.stripMargin)
              }
            }
          } else {
            initDomainDf
          }

          // If `domainSpaceDf` has any row with an empty domain, fills it with a default domain.
          val domainDf = if (corrAttrsWithScores.isEmpty || domainSpaceDf.where("size(domain) = 0").count() > 0) {
            logWarning(s"Empty domains found in '$attrName', so fills them with default ones")

            withTempView(domainSpaceDf) { domainView =>
              // Since any correlated attribute not found, we need to select domains randomly
              sparkSession.sql(
                s"""
                   |SELECT
                   |  $rvId,
                   |  $rowId,
                   |  attrName,
                   |  initValue,
                   |  IF(size(domain) > 0, domain, defaultDomain) domain
                   |FROM
                   |  $domainView, (
                   |    SELECT collect_set($attrName) defaultDomain
                   |    FROM (
                   |      SELECT *
                   |      FROM $attrStatView
                   |      WHERE ${whereCaluseToFilterStat(attrName)}
                   |      ORDER BY cnt DESC
                   |      LIMIT $defaultMaxDomainSize
                   |    )
                   |  )
                 """.stripMargin)
            }
          } else {
            domainSpaceDf
          }

          withTempView(domainDf) { domainView =>
            // Computes domains for error cells
            val ftAttrToId = featureAttrs.zipWithIndex.toMap
            sparkSession.sql(
              s"""
                 |SELECT
                 |  $rvId,
                 |  $rowId,
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
                 |    $rvId,
                 |    $rowId,
                 |    attrName,
                 |    array_sort(array_union(array(initValue), domain)) domain,
                 |    IF(ISNULL(initValue), shuffle(domain)[0], initValue) initValue
                 |  FROM
                 |    $domainView
                 |)
               """.stripMargin)
          }
        }
      }.reduce(_.union(_))

      val cellDomainView = createAndCacheTempView(cellDomainDf, "cell_domain")
      metadata.add("cell_domain", cellDomainView)

      // Number of rows in `cellDomainView` is the same with the number of error cells
      assert(cellDomainDf.count == sparkSession.table(errCellView).count)

      val weakLabelDf = sparkSession.sql(
        s"""
           |SELECT
           |  $rvId, weakLabel, weakLabelIndex, fixed, IF(domainSize > 1, false, true) AS clean
           |FROM
           |  $cellDomainView AS t1
           |LEFT OUTER JOIN
           |  $errCellView AS t2
           |ON
           |  t1.$rowId = t2.$rowId AND
           |  t1.attrName = t2.attrName
           |WHERE
           |  weakLabel IS NOT NULL AND
           |  t1.fixed != 1
         """.stripMargin)

      metadata.add("weak_labels", createAndCacheTempView(weakLabelDf, "weak_labels"))

      val varMaskDf = sparkSession.sql(s"SELECT $rvId, domainSize FROM $cellDomainView")
      metadata.add("var_masks", createAndCacheTempView(varMaskDf, "var_masks"))

      val (totalVars, classes) = sparkSession.sql(s"SELECT COUNT($rvId), MAX(domainSize) FROM $cellDomainView")
        .collect.headOption.map { case Row(l: Long, i: Int) => (l, i) }.get

      logBasedOnLevel(s"totalVars=$totalVars classes=$classes " +
        s"featureAttrs(${featureAttrs.size})=${featureAttrs.mkString(",")}")
      metadata.add("total_vars", s"$totalVars")
      metadata.add("feature_attrs", featureAttrs)
      metadata.add("classes", s"$classes")

      val posValDf = sparkSession.sql(
        s"""
           |SELECT $rvId, $rowId, attrName, posexplode(domain) (valId, rVal)
           |FROM $cellDomainView
         """.stripMargin)

      metadata.add("pos_values", createAndCacheTempView(posValDf, "pos_values"))
      metadata.toJson
    }
  }
}
