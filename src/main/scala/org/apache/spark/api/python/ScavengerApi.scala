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

import java.io.{File, FileWriter}
import java.sql.{Connection, Statement}

import org.apache.spark.SparkException
import org.apache.spark.python._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.types._
import org.apache.spark.util.ScavengerUtils._
import org.apache.spark.util.{Utils => SparkUtils}

/** A Python API entry point to use the Scavenger functionality. */
object ScavengerApi extends BaseSchemaSpyApi {

  private def dumpSparkCatalog(spark: SparkSession, tables: Seq[TableIdentifier]): File = {
    val dbFile = new File(SparkUtils.createTempDir(namePrefix = "scavenger"), "generated.db")
    val connUrl = s"jdbc:sqlite:${dbFile.getAbsolutePath}"

    var conn: Connection = null
    var stmt: Statement = null
    try {
      conn = getDriver(getSqliteDriverName(), "org.sqlite.JDBC").connect(connUrl, emptyProp)
      stmt = conn.createStatement()

      def toSQLiteTypeName(dataType: DataType): Option[String] = dataType match {
        case IntegerType => Some("INTEGER")
        case StringType => Some("TEXT")
        case _ => None
      }

      tables.foreach { t =>
        val schema = spark.table(t.quotedString).schema
        val attrDefs = schema.flatMap { f => toSQLiteTypeName(f.dataType).map(tpe => s"${f.name} $tpe") }
        if (attrDefs.nonEmpty) {
          val ddlStr =
            s"""
               |CREATE TABLE ${t.identifier} (
               |  ${attrDefs.mkString(",\n")}
               |);
             """.stripMargin

          logDebug(
            s"""
               |SQLite DDL String from a Spark schema: `${schema.toDDL}`:
               |$ddlStr
             """.stripMargin)

          stmt.execute(ddlStr)
        }
      }
    } finally {
      if (stmt != null) stmt.close()
      if (conn != null) conn.close()
    }
    dbFile
  }

  // To call this function, just say a line below;
  // >>> scavenger.setDbName('default').setInferType('basic').infer().show()
  def infer(userDefinedOutputPath: String, dbName: String, inferType: String): String = {
    withSparkSession { sparkSession =>
      val fkInferType = inferType match {
        case "schema-only" => FkInferType.SCHEMA_ONLY
        case "default" | "basic" | "base" => FkInferType.BASIC
        case "noop" => FkInferType.NOOP
        case _ => throw new IllegalArgumentException(s"Unsupported inferType: $inferType")
      }

      val tables = sparkSession.sessionState.catalog.listTables(dbName)

      val outputPath = if (userDefinedOutputPath.nonEmpty) {
        userDefinedOutputPath
      } else {
        getTempOutputPath()
      }

      val propFile = generatePropFile {
        val jdbcDriverFile = getJdbcDriverFile(getSqliteDriverName())
        val dbFile = dumpSparkCatalog(sparkSession, tables)
        (jdbcDriverFile.getAbsolutePath, s"schemaspy.db=${dbFile.getAbsolutePath}")
      }

      runSchemaSpy("sqlite-xerial", outputPath, propFile, Some(_ => {
        generateMetaFile {
          val fkConstraints = FkInference(sparkSession, fkInferType).infer(tables)
          fkConstraints.map { case (table, fkDefs) =>
            s"""
               |<table name="$table" comments="">
               |  ${fkDefs.map { fk =>
                      s"""
                         |<column name="${fk._1}" comments="">
                         |  <foreignKey table="${fk._2._1}" column="${fk._2._2}" />
                         |</column>
                       """.stripMargin
                  }.mkString("\n")}
               |</table>
             """.stripMargin
          }.mkString("\n")
        }}))
    }
  }

  // To call this function, just say a line below;
  // >>> scavenger.constraints().setDbName('default').setTableName('t').infer().show()
  def inferConstraints(userDefinedOutputPath: String, dbName: String, tableName: String): String = {
    withSparkSession { sparkSession =>
      val table = sparkSession.sessionState.catalog.listTables(dbName)
        .find(_.table == tableName).getOrElse {
          throw new SparkException(s"Table '$tableName' does not exist in database '$dbName'.")
        }

      val outputPath = if (userDefinedOutputPath.nonEmpty) {
        userDefinedOutputPath
      } else {
        getTempOutputPath()
      }

      val propFile = generatePropFile {
        val jdbcDriverFile = getJdbcDriverFile(getSqliteDriverName())
        val dbFile = dumpSparkCatalog(sparkSession, table :: Nil)
        (jdbcDriverFile.getAbsolutePath, s"schemaspy.db=${dbFile.getAbsolutePath}")
      }

      runSchemaSpy("sqlite-xerial", outputPath, propFile, Some(_ => {
        generateMetaFile {
          val tck = IntegrityConstraintDiscovery.tableConstraintsKey
          val constraints = IntegrityConstraintDiscovery.exec(sparkSession, table.table)
          val (tableConstraints, columnConstraints) = constraints
            .map { case (k, v) => k -> v.map(s => s.replace("&", "&amp;")).mkString("&lt;br&gt;") }
            .partition { v => v._1 == tck }

          // Writes constraints in a text file
          constraints.get(IntegrityConstraintDiscovery.tableConstraintsKey).foreach { cs =>
            var fileWriter: FileWriter = null
            try {
              val constraintFile = new File(outputPath, "constraints.txt")
              fileWriter = new FileWriter(constraintFile)
              fileWriter.write(cs.mkString("\n"))
            } finally {
              if (fileWriter != null) {
                fileWriter.close()
              }
            }
          }

          val tableComment = tableConstraints.values.headOption.getOrElse("")
          s"""
             |<table name="$tableName" comments="$tableComment">
             |  ${columnConstraints.filterNot(_._1 == tck).map { case (col, columnComment) =>
                  s"""
                     |<column name="$col" comments="$columnComment" />
                   """.stripMargin
                }.mkString("\n")}
             |</table>
             """.stripMargin
        }}))
    }
  }
}
