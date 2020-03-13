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

import org.apache.spark.python._

import io.github.maropu.Utils._

/** A Python API entry point to create PyTorch features for data cleaning. */
object ScavengerRepairFeatureApi extends BaseScavengerRepairApi {

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
      if (constraints.entries.isEmpty) {
        // An empty string just means that any valid constraint is not found
        ""
      } else {
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

        withTempView(errRowDf, cache = true) { errRowView =>
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
            outputToConsole(s"Starts processing the $i/${queries.size} query to compute #violations...")
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
}
