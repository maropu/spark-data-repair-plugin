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

import java.io.File
import java.net.URI
import java.util.Locale
import java.util.concurrent.atomic.AtomicInteger

import scala.collection.mutable
import scala.io.Source
import scala.sys.process.{Process, ProcessLogger}
import scala.util.Try

import org.apache.commons.io.FileUtils

import org.apache.spark.SparkException
import org.apache.spark.python.DenialConstraints
import org.apache.spark.sql.ExceptionUtils.AnalysisException
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.util.stringToFile
import org.apache.spark.sql.types.StringType
import org.apache.spark.util.BlockingLineStream
import org.apache.spark.util.RepairUtils.{getRandomString, withTempView}

private[python] object DepGraph extends RepairBase {

  private def generateGraphString(nodes: Seq[String], edges: Seq[String]) = {
    if (nodes.nonEmpty) {
      s"""
         |digraph {
         |  graph [pad="0.5" nodesep="1.0" ranksep="4" fontname="Helvetica" rankdir=LR];
         |  node [shape=plaintext]
         |
         |  ${nodes.sorted.mkString("\n")}
         |  ${edges.sorted.mkString("\n")}
         |}
       """.stripMargin
    } else {
      throw new SparkException("Failed to a generate dependency graph because " +
        "no correlated attribute found")
    }
  }

  private def normalizeForHtml(str: String) = {
    str.replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
  }

  private def trimString(s: String, maxLength: Int): String = {
    if (s.length > maxLength) s"${s.substring(0, maxLength)}..." else s
  }

  private def generateNodeString(
      nodeName: String,
      valuesWithIndex: Seq[(String, Int)],
      maxStringLength: Int) = {
    val entries = valuesWithIndex.map { case (v, i) =>
      s"""<tr><td port="$i">${normalizeForHtml(trimString(v, maxStringLength))}</td></tr>"""
    }
    s"""
       |"$nodeName" [color="black" label=<
       |  <table>
       |    <tr><td bgcolor="black" port="nodeName"><i><font color="white">$nodeName</font></i></td></tr>
       |    ${entries.mkString("\n")}
       |  </table>>];
     """.stripMargin
  }

  private val nextNodeId = new AtomicInteger(0)

  def computeDepGraph(
      inputView: String,
      targetAttrs: Seq[String],
      maxDomainSize: Int,
      maxAttrValueNum: Int,
      maxAttrValueLength: Int,
      samplingRatio: Double,
      minCorrThres: Double,
      edgeLabel: Boolean): String = {
    assert(targetAttrs.nonEmpty)

    val domainStatMap = {
      val statMap = RepairApi.computeAndGetTableStats(inputView).mapValues(_.distinctCount)
      val targetAttrSet = targetAttrs.toSet
      statMap.filter { case (attr, stat) =>
        targetAttrSet.contains(attr) && stat <= maxDomainSize
      }
    }

    if (domainStatMap.size < 2) {
      throw AnalysisException("At least two candidate attributes needed to " +
        "build a dependency graph")
    }

    val attrPairsToComputeDeps = domainStatMap.keys.toSeq.combinations(2).map {
      case p @ Seq(x, y) if domainStatMap(x) < domainStatMap(y) => p.reverse
      case p => p
    }.toSeq

    val rowCount = spark.table(inputView).count()
    val attrSetToComputeFreqStats = domainStatMap.keys.map(k => Seq(k)).toSeq ++ attrPairsToComputeDeps
    val attrFreqStatDf = RepairApi.computeFreqStats(
      inputView, attrSetToComputeFreqStats, samplingRatio, 0.0)

    val hubNodes = mutable.ArrayBuffer[(String, String)]()
    val nodeDefs = mutable.ArrayBuffer[String]()
    val edgeDefs = mutable.ArrayBuffer[String]()

    withTempView(attrFreqStatDf, "freq_attr_stats", cache = true) { attrStatView =>
      val targetAttrs = domainStatMap.keys.toSeq
      val targetAttrPairs = attrPairsToComputeDeps.map { case Seq(x, y) => (x, y) }
      val pairwiseStatMap = RepairApi.computePairwiseStats(
        inputView, rowCount, attrStatView, targetAttrs, targetAttrPairs, domainStatMap)

      val attrPairs = attrPairsToComputeDeps.filter { case Seq(x, y) =>
        pairwiseStatMap(x).exists { case (attr, h) =>
          y == attr && Math.max(h, 0.0) >= minCorrThres
        }
      }

      if (attrPairs.isEmpty) {
        throw AnalysisException("No highly-correlated attribute pair " +
          s"(threshold: $minCorrThres) found")
      }

      attrPairs.foreach { case Seq(x, y) =>
        val df = spark.sql(
          s"""
             |SELECT CAST(`$x` AS STRING) x, collect_set(named_struct('y', CAST(`$y` AS STRING), 'cnt', cnt)) ys
             |FROM $attrStatView
             |WHERE `$x` IS NOT NULL AND `$y` IS NOT NULL
             |GROUP BY $x
           """.stripMargin)

        val rows = df.collect()
        val truncate = maxAttrValueNum < rows.length
        val edgeCands = rows.take(maxAttrValueNum).map { case Row(xv: String, ys: Seq[Row]) =>
          val yvs = ys.map { case Row(y: String, cnt: Long) => (y, cnt) }
          (xv, yvs)
        }

        def genNode(nodeName: String, values: Seq[String]): (String, Map[String, Int]) = {
          val nn = s"${nodeName}_${nextNodeId.getAndIncrement()}"
          val valuesWithIndex = {
            if (truncate) values.zipWithIndex :+ ("...", -1) else values.zipWithIndex
          }
          hubNodes += ((nn, nodeName))
          nodeDefs += generateNodeString(nn, valuesWithIndex, maxAttrValueLength)
          (nn, valuesWithIndex.toMap)
        }
        if (edgeCands.nonEmpty) {
          val (xNodeName, valueToIndexMapX) = genNode(x, edgeCands.map(_._1))
          val (yNodeName, valueToIndexMapY) = genNode(y, edgeCands.flatMap(_._2.map(_._1)).distinct)

          def genEdge(from: String, to: String, cnt: Long, totalCnt: Long, label: Boolean): String = {
            val p = (cnt + 0.0) / totalCnt
            val w = 0.1 + Math.log(cnt) / (0.1 + Math.log((rowCount + 0.0) / valueToIndexMapX.size))
            val c = s"gray${(100.0 * (1.0 - p)).toInt}"
            val labelOpt = if (label) s"""label="$cnt/$totalCnt"""" else ""
            s""""$xNodeName":${valueToIndexMapX(from)} -> "$yNodeName":${valueToIndexMapY(to)} """ +
              s"""[ color="$c" penwidth="$w" $labelOpt ];"""
          }

          edgeCands.foreach { case (xv: String, yvs: Seq[(String, Long)]) =>
            val totalCnt = yvs.map(_._2).sum
            yvs.foreach { case (yv, cnt) =>
              edgeDefs += genEdge(xv, yv, cnt, totalCnt, label = edgeLabel)
            }
          }
        }
      }
    }

    // Add entries for hub nodes
    hubNodes.foreach { case (n, h) =>
      nodeDefs += s""""$h" [ shape="box" ];"""
      edgeDefs += s""""$h" -> "$n":nodeName [ arrowhead="diamond" penwidth="1.0" ];"""
    }

    generateGraphString(nodeDefs, edgeDefs)
  }

  val validImageFormatSet = Set("png", "svg")

  private def isCommandAvailable(command: String): Boolean = {
    val attempt = {
      Try(Process(Seq("sh", "-c", s"command -v $command")).run(ProcessLogger(_ => ())).exitValue())
    }
    attempt.isSuccess && attempt.get == 0
  }

  // If the Graphviz dot command installed, converts the generated dot file
  // into a specified-formatted image.
  private def tryGenerateImageFile(format: String, src: String, dst: String): Unit = {
    if (isCommandAvailable("dot")) {
      try {
        val commands = Seq("bash", "-c", s"dot -T$format $src > $dst")
        BlockingLineStream(commands)
      } catch {
        case _: Throwable =>
          logWarning("Cannot generate image file because `dot` command not installed.")
      }
    }
  }

  def generateDepGraph(
      outputDirPath: String,
      inputView: String,
      format: String,
      targetAttrs: Seq[String],
      maxDomainSize: Int,
      maxAttrValueNum: Int,
      maxAttrValueLength: Int,
      samplingRatio: Double,
      minCorrThres: Double,
      edgeLabel: Boolean,
      filenamePrefix: String,
      overwrite: Boolean): Unit = {
    val graphString = computeDepGraph(
      inputView, targetAttrs, maxDomainSize, maxAttrValueNum, maxAttrValueLength,
      samplingRatio, minCorrThres, edgeLabel)
    if (!validImageFormatSet.contains(format.toLowerCase(Locale.ROOT))) {
      throw AnalysisException(s"Invalid image format: $format")
    }
    val outputDir = new File(outputDirPath)
    if (overwrite) {
      FileUtils.deleteDirectory(outputDir)
    }
    if (!outputDir.mkdir()) {
      throw AnalysisException(if (overwrite) {
        s"`overwrite` is set to true, but could not remove output dir path '$outputDirPath'"
      } else {
        s"output dir path '$outputDirPath' already exists"
      })
    }
    val dotFile = stringToFile(new File(outputDir, s"$filenamePrefix.dot"), graphString)
    val srcFile = dotFile.getAbsolutePath
    val dstFile = new File(outputDir, s"$filenamePrefix.$format").getAbsolutePath
    tryGenerateImageFile(format, srcFile, dstFile)
  }

  def computeFunctionalDeps(inputView: String, constraintFilePath: String, targetAttrs: Seq[String]): String = {
    val (inputDf, qualifiedName) = checkAndGetQualifiedInputName("", inputView)
    var file: Source = null
    val constraints = try {
      file = Source.fromFile(new URI(constraintFilePath).getPath)
      file.getLines()
      DenialConstraints.parseAndVerifyConstraints(file.getLines(), qualifiedName, inputDf.columns)
    } finally {
      if (file != null) {
        file.close()
      }
    }

    // TODO: Reuse the previous computation result
    val domainSizes = {
      // TODO: `StringType` only supported now
      val supported = inputDf.schema.filter(_.dataType == StringType).map(_.name).toSet
      RepairApi.computeAndGetTableStats(inputView).mapValues(_.distinctCount)
        .filter(kv => supported.contains(kv._1))
    }

    val fdMap = mutable.Map[String, mutable.Set[String]]()

    def hasNoCyclic(ref1: String, ref2: String): Boolean = {
      !fdMap.get(ref1).exists(_.contains(ref2)) && !fdMap.get(ref2).exists(_.contains(ref1))
    }

    constraints.predicates.filter { preds =>
      // Filters predicate candidates that might mean functional deps
      preds.length == 2 && preds.flatMap(_.references).distinct.length == 2
    }.foreach {
      case Seq(p1, p2) if Set(p1.sign, p2.sign) == Set("EQ", "IQ") &&
        p1.references.length == 1 && p2.references.length == 1 =>
        val ref1 = p1.references.head
        val ref2 = p2.references.head
        (domainSizes.get(ref1), domainSizes.get(ref2)) match {
          case (Some(ds1), Some(ds2)) if ds1 < ds2 =>
            fdMap.getOrElseUpdate(ref1, mutable.Set[String]()) += ref2
          case (Some(ds1), Some(ds2)) if ds1 > ds2 =>
            fdMap.getOrElseUpdate(ref2, mutable.Set[String]()) += ref1
          case (Some(_), Some(_)) if hasNoCyclic(ref1, ref2) =>
            fdMap(ref1) = mutable.Set(ref2)
          case _ =>
        }
      case _ =>
    }

    // TODO: We need a smarter way to convert Scala data to a json string
    fdMap.filterKeys(targetAttrs.contains).map { case (k, values) =>
      s""""$k": [${values.toSeq.sorted.map { v => s""""$v"""" }.mkString(",")}]"""
    }.mkString("{", ",", "}")
  }

  def computeFunctionalDepMap(inputView: String, X: String, Y: String): String = {
    val x = getRandomString(prefix="x")
    val y = getRandomString(prefix="y")
    val df = spark.sql(
      s"""
         |SELECT CAST($X AS STRING) $x, CAST($y[0] AS STRING) $y FROM (
         |  SELECT $X, collect_set($Y) $y
         |  FROM $inputView
         |  GROUP BY $X
         |  HAVING size($y) = 1
         |)
       """.stripMargin)

    // TODO: We need a smarter way to convert Scala data to a json string
    df.collect.map { case Row(x: String, y: String) =>
      s""""$x": "$y""""
    }.mkString("{", ",", "}")
  }
}
