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

import java.io.{File, PrintWriter}
import java.sql.DriverManager
import java.util.UUID

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.{FkInferType, FkInference}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.util.Utils

import io.github.maropu.SchemaSpyLauncher

/** An Python API entry point to call a SchemaSpy command. */
object ScavengerApi extends Logging {

  private val JDBC_DRIVERS_HOME = "JDBC_DRIVERS_HOME"
  private val JDBC_SQLITE_VERSION = "JDBC_SQLITE_VERSION"
  private val JDBC_POSTGRESQL_VERSION = "JDBC_POSTGRESQL_VERSION"

  private def withSparkSession(f: SparkSession => String): String = {
    SparkSession.getActiveSession.map { sparkSession =>
      f(sparkSession)
    }.getOrElse {
      throw new SparkException("An active SparkSession not found.")
    }
  }

  private def getEnvOrFail(key: String): String = {
    val value = System.getenv(key)
    if (value == null) {
      throw new SparkException(s"'$key' not defined correctly.")
    }
    value
  }

  private def getTempOutputPath(): String = {
    s"${System.getProperty("java.io.tmpdir")}/schemaspy-${UUID.randomUUID.toString}.output"
  }

  // To call this function, just say a lien below;
  // >>> spySchema('-dbhelp')
  def run(args: String): Unit = {
    SchemaSpyLauncher.run(args.split(" "): _*)
  }

  // To call this function, just say a lien below;
  // >>> schemaspy.setDbName('postgres').setDriverName('postgresql').setProps('host=localhost,port=5432').run().show()
  def run(userDefinedOutputPath: String, dbName: String, driverName: String, props: String): String = {
    val outputPath = if (userDefinedOutputPath.isEmpty) {
      getTempOutputPath()
    } else {
      userDefinedOutputPath
    }

    val tempDir = Utils.createTempDir(namePrefix = "schemaspy")
    val jdbcDriversHome = getEnvOrFail(JDBC_DRIVERS_HOME)
    val (driverType, jdbcDriverName, dbSpecificProps) = driverName match {
      case "sqlite" =>
        val jdbcVersion = getEnvOrFail(JDBC_SQLITE_VERSION)
        ("sqlite-xerial",
          s"sqlite-jdbc-$jdbcVersion.jar",
          s"""
             |schemaspy.db=$dbName
           """.stripMargin)
      case "postgresql" =>
        val jdbcVersion = getEnvOrFail(JDBC_POSTGRESQL_VERSION)
        val propsMap = props.split(",").map { prop => prop.split("=").toSeq }.map {
          case Seq(k, v) => k -> v
          case prop => throw new SparkException(s"Illegal property format: ${prop.mkString(",")}")
        }.toMap
        ("pgsql",
          s"postgresql-$jdbcVersion.jar",
          s"""
             |schemaspy.db=$dbName
             |schemaspy.host=${propsMap.getOrElse("host", "")}
             |schemaspy.port=${propsMap.getOrElse("port", "")}
           """.stripMargin)
      case _ =>
        throw new SparkException(s"Unknown JDBC driver: $driverName")
    }
    val jdbcDriverPath = s"$jdbcDriversHome/$jdbcDriverName"
    if (!new File(jdbcDriverPath).exists()) {
      throw new SparkException(s"'$jdbcDriverName' does not exist in $jdbcDriversHome.")
    }

    val propFile = new File(tempDir, "schemaspy.properties")
    val propWriter = new PrintWriter(propFile, "UTF-8")
    try {
      propWriter.write(
        s"""
           |schemaspy.dp=$jdbcDriverPath
           |$dbSpecificProps
           |schemaspy.u=${System.getProperty("user.name")}
           |schemaspy.s=%
       """.stripMargin)
    } finally {
      propWriter.close()
    }

    SchemaSpyLauncher.run(Seq(
      "-configFile", propFile.getAbsolutePath,
      "-t", driverType,
      "-cat", "%",
      "-o", outputPath
    ): _*)

    outputPath
  }

  // To call this function, just say a lien below;
  // >>> schemaspy.setDbName('default').setInferType('basic').infer().show()
  def infer(userDefinedOutputPath: String, dbName: String, inferType: String): String = {
    withSparkSession { sparkSession =>
      val fkInferType = inferType match {
        case "schema-only" => FkInferType.SCHEMA_ONLY
        case "default" | "basic" | "base" => FkInferType.BASIC
        case "noop" => FkInferType.NOOP
        case _ => throw new IllegalArgumentException(s"Unsupported inferType: $inferType")
      }

      val outputPath = if (userDefinedOutputPath.isEmpty) {
        getTempOutputPath()
      } else {
        userDefinedOutputPath
      }

      val tempDir = Utils.createTempDir(namePrefix = "schemaspy")
      val jdbcDriversHome = getEnvOrFail(JDBC_DRIVERS_HOME)
      val jdbcVersion = getEnvOrFail(JDBC_SQLITE_VERSION)
      val jdbcDriverName = s"sqlite-jdbc-$jdbcVersion.jar"
      val jdbcDriverPath = s"$jdbcDriversHome/$jdbcDriverName"
      if (!new File(jdbcDriverPath).exists()) {
        throw new SparkException(s"'$jdbcDriverName' does not exist in $jdbcDriverPath.")
      }

      val tables = sparkSession.sessionState.catalog.listTables(dbName)

      val dbPath = {
        Class.forName("org.sqlite.JDBC")
        val dbFile = new File(tempDir, "generated.db")
        val conn = DriverManager.getConnection(s"jdbc:sqlite:${dbFile.getAbsolutePath}")
        val stmt = conn.createStatement()

        def toSQLiteTypeName(dataType: DataType): Option[String] = dataType match {
          case IntegerType => Some("INTEGER")
          case StringType => Some("TEXT")
          case _ => None
        }

        tables.foreach { t =>
          val schema = sparkSession.table(t.quotedString).schema
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
        conn.close()
        dbFile.getAbsolutePath
      }

      val propFile = new File(tempDir, "schemaspy.properties")
      val propWriter = new PrintWriter(propFile, "UTF-8")
      try {
        propWriter.write(
          s"""
             |schemaspy.dp=$jdbcDriverPath
             |schemaspy.db=$dbPath
             |schemaspy.u=${System.getProperty("user.name")}
             |schemaspy.s=%
         """.stripMargin)
      } finally {
        propWriter.close()
      }

      val metaFile = new File(tempDir, "meta.xml")
      val metaWriter = new PrintWriter(metaFile, "UTF-8")
      try {
        val fkConstraints = FkInference(sparkSession, fkInferType).infer(tables)
        metaWriter.write(
          s"""
             |<schemaMeta xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://schemaspy.org/xsd/6/schemameta.xsd" >
             |  <comments>AUTO-GENERATED</comments>
             |  <tables>
             |    ${fkConstraints.map { case (table, fkDefs) =>
                    s"""
                       |<table name="$table">
                       |  ${fkDefs.map { fk =>
                            s"""
                               |<column name="${fk._1}">
                               |  <foreignKey table="${fk._2._1}" column="${fk._2._2}" />
                               |</column>
                             """.stripMargin
                          }.mkString("\n")}
                       |</table>
                     """.stripMargin
                }.mkString("\n")}
             |  </tables>
             |</schemaMeta>
           """.stripMargin)
      } finally {
        metaWriter.close()
      }

      SchemaSpyLauncher.run(Seq(
        "-configFile", propFile.getAbsolutePath,
        "-meta", metaFile.getAbsolutePath,
        "-t", "sqlite-xerial",
        "-cat", "%",
        "-o", outputPath
      ): _*)

      outputPath
    }
  }
}
