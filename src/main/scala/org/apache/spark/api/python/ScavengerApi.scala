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
import java.net.URL
import java.sql.{Connection, Driver, ResultSet, Statement}
import java.util.UUID

import scala.collection.mutable

import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.{FkInferType, FkInference, IntegrityConstraintDiscovery}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.types._
import org.apache.spark.util.{MutableURLClassLoader, Utils}

import io.github.maropu.SchemaSpyLauncher

/** A Python API entry point to call a SchemaSpy command. */
object ScavengerApi extends Logging {

  private val JDBC_DRIVERS_HOME = "JDBC_DRIVERS_HOME"
  private val JDBC_SQLITE_VERSION = "JDBC_SQLITE_VERSION"
  private val JDBC_POSTGRESQL_VERSION = "JDBC_POSTGRESQL_VERSION"

  private val emptyProp = new java.util.Properties()

  private def withSparkSession[T](f: SparkSession => T): T = {
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
    s"${System.getProperty("java.io.tmpdir")}/scavenger-${UUID.randomUUID.toString}.output"
  }

  private def schemaSpyMetaTemplate(tableContent: String): String = {
    s"""
       |<schemaMeta xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://schemaspy.org/xsd/6/schemameta.xsd" >
       |  <comments>AUTO-GENERATED</comments>
       |  <tables>
       |    $tableContent
       |  </tables>
       |</schemaMeta>
     """.stripMargin
  }

  private def getSqliteDriverName(): String = {
    s"sqlite-jdbc-${getEnvOrFail(JDBC_SQLITE_VERSION)}.jar"
  }

  private def getPostgresqlDriverName(): String = {
    s"postgresql-${getEnvOrFail(JDBC_POSTGRESQL_VERSION)}.jar"
  }

  private def getJdbcDriverFile(driverName: String): File = {
    val driversHome = getEnvOrFail(JDBC_DRIVERS_HOME)
    val driverPath = s"$driversHome/$driverName"
    val driverFile = new File(driverPath)
    if (!driverFile.exists()) {
      throw new SparkException(s"'$driverName' does not exist in $driverPath.")
    }
    driverFile
  }

  private def dumpSparkCatalog(spark: SparkSession, tables: Seq[TableIdentifier]): File = {
    val dbFile = new File(Utils.createTempDir(namePrefix = "scavenger"), "generated.db")
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

  private def generatePropFile(f: => (String, String)): File = {
    val tempDir = Utils.createTempDir(namePrefix = "scavenger")
    val propFile = new File(tempDir, "schemaspy.properties")
    val propWriter = new PrintWriter(propFile, "UTF-8")
    try {
      val (jdbcDriverPath, dbSpecificProps) = f
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
    propFile
  }

  private def generateMetaFile(tableContent: => String): File = {
    val tempDir = Utils.createTempDir(namePrefix = "scavenger")
    val metaFile = new File(tempDir, "meta.xml")
    val metaWriter = new PrintWriter(metaFile, "UTF-8")
    try {
      metaWriter.write(schemaSpyMetaTemplate(tableContent))
    } finally {
      metaWriter.close()
    }
    metaFile
  }

  // To call this function, just say a lien below;
  // >>> spySchema('-dbhelp')
  def run(args: String): Unit = {
    SchemaSpyLauncher.run(args.split(" "): _*)
  }

  private def runSchemaSpy(
      jdbcDriverTypeName: String,
      userDefinedOutputPath: String,
      propFile: File,
      doProfileFunc: Option[Unit => File] = None): String = {

    val additionalConfigs = doProfileFunc.map { f =>
      val metaFile = f()
      Seq("-meta", metaFile.getAbsolutePath)
    }.getOrElse {
      Seq.empty
    }

    val outputPath = if (userDefinedOutputPath.nonEmpty) {
      userDefinedOutputPath
    } else {
      getTempOutputPath()
    }

    SchemaSpyLauncher.run(Seq(
      "-configFile", propFile.getAbsolutePath,
      "-t", jdbcDriverTypeName,
      "-cat", "%",
      "-o", outputPath
    ) ++ additionalConfigs: _*)

    outputPath
  }

  private def parseStringPropsAsMap(props: String):Map[String, String] = {
    props.split(",").map { prop => prop.split("=").toSeq }.map {
      case Seq(k, v) => k -> v
      case prop => throw new SparkException(s"Illegal property format: ${prop.mkString(",")}")
    }.toMap
  }

  private def getTypeNameString(jdbcTypeId: Int): String = jdbcTypeId match {
    // scalastyle:off
    case java.sql.Types.BIGINT                  => "bigint"
    case java.sql.Types.BINARY                  => "binary"
    case java.sql.Types.BIT                     => "bit"
    case java.sql.Types.BLOB                    => "blob"
    case java.sql.Types.BOOLEAN                 => "boolean"
    case java.sql.Types.CHAR                    => "char"
    case java.sql.Types.CLOB                    => "clob"
    case java.sql.Types.DATE                    => "date"
    case java.sql.Types.DECIMAL                 => "decimal"
    case java.sql.Types.DOUBLE                  => "double"
    case java.sql.Types.FLOAT                   => "float"
    case java.sql.Types.INTEGER                 => "int"
    case java.sql.Types.NULL                    => "null"
    case java.sql.Types.NUMERIC                 => "numeric"
    case java.sql.Types.REAL                    => "real"
    case java.sql.Types.SMALLINT                => "smallint"
    case java.sql.Types.TIME                    => "time"
    case java.sql.Types.TIME_WITH_TIMEZONE      => "time"
    case java.sql.Types.TIMESTAMP               => "timestamp"
    case java.sql.Types.TIMESTAMP_WITH_TIMEZONE => "timestamp"
    case java.sql.Types.TINYINT                 => "tinyint"
    case java.sql.Types.VARCHAR                 => "varchar"
    case _                                      => "unknown"
    // scalastyle:on
  }

  // Loads a JDBC jar file on runtime and returns an implementation-specific `Driver`
  private def getDriver(jdbcDriverName: String, className: String): Driver = {
    val contextClassLoader = Thread.currentThread.getContextClassLoader
    val classLoader = new MutableURLClassLoader(new Array[URL](0), contextClassLoader)
    val jdbcDriverFile = getJdbcDriverFile(jdbcDriverName)
    classLoader.addURL(jdbcDriverFile.toURI.toURL)
    classLoader.loadClass(className).newInstance().asInstanceOf[Driver]
  }

  // To call this function, just say a lien below;
  // >>> schemaspy.setDbName('postgres').setDriverName('postgresql').setProps('host=localhost,port=5432').catalogToDataFrame()
  def catalogToDataFrame(dbName: String, driverName: String, props: String): DataFrame = {

    def processResultSet(rs: ResultSet)(f: ResultSet => Unit): Unit = {
      try { while (rs.next()) { f(rs) } } finally { rs.close() }
    }

    withSparkSession { sparkSession =>
      // Parses input `props as `Map`
      val propsMap = parseStringPropsAsMap(props)

      val (driverClassName, jdbcDriverName, connUrl) = driverName match {
        case "postgresql" =>
          ("org.postgresql.Driver", getPostgresqlDriverName(),
            s"jdbc:postgresql://${propsMap("host")}:${propsMap("port")}/$dbName")
        case _ =>
          throw new SparkException(s"Unknown JDBC driver: $driverName")
      }

      var conn: Connection = null
      try {
        conn = getDriver(jdbcDriverName, driverClassName).connect(connUrl, emptyProp)
        val tables = mutable.ArrayBuffer[String]()
        val columnNames = Seq("tableName", "columnName", "type", "nullable", "isPrimaryKey", "isForeignKey")
        val columns = mutable.ArrayBuffer[(String, String, String, Boolean, Boolean, Boolean)]()
        val meta = conn.getMetaData
        processResultSet(meta.getTables(dbName, null, "%", Array("TABLE", "VIEW"))) { rs =>
          tables.append(rs.getString("TABLE_NAME"))
        }

        // First, collects information about primary/foreign keys
        val pkFkMap = tables.flatMap { t =>
          val buf = mutable.ArrayBuffer[((String, String), String)]()
          processResultSet(meta.getPrimaryKeys(null, null, t)) { rs =>
            buf.append(((rs.getString("TABLE_NAME"), rs.getString("COLUMN_NAME")), "PK"))
          }
          processResultSet(meta.getImportedKeys(null, null, t)) { rs =>
            buf.append(((rs.getString("FKTABLE_NAME"), rs.getString("FKCOLUMN_NAME")), "FK"))
          }
          buf
        }.toMap

        // Then, dumps catalog table entires for `dbName`
        tables.foreach { t =>
          processResultSet(meta.getColumns(dbName, null, t, "%")) { rs =>
            val tableName = rs.getString("TABLE_NAME")
            val columnName = rs.getString("COLUMN_NAME")
            val tpe = getTypeNameString(rs.getShort("DATA_TYPE"))
            val nullable = rs.getShort("NULLABLE") == 0
            val pkFkOpt = pkFkMap.get((tableName, columnName))
            val isPk = pkFkOpt.contains("PK")
            val isFk = pkFkOpt.contains("FK")
            columns.append((tableName, columnName, tpe, nullable, isPk, isFk))
          }
        }
        import sparkSession.implicits._
        columns.toDF(columnNames: _*)
      } finally {
        if (conn != null) {
          conn.close()
        }
      }
    }
  }

  // To call this function, just say a lien below;
  // >>> schemaspy.setDbName('postgres').setDriverName('postgresql').setProps('host=localhost,port=5432').run().show()
  def run(userDefinedOutputPath: String, dbName: String, driverName: String, props: String): String = {
    // Parses input `props as `Map`
    val propsMap = parseStringPropsAsMap(props)

    val (driverTypeName, jdbcDriverName, dbSpecificProps) = driverName match {
      case "sqlite" =>
        ("sqlite-xerial",
          getSqliteDriverName(),
          s"""
             |schemaspy.db=$dbName
           """.stripMargin)
      case "postgresql" =>
        ("pgsql",
          getPostgresqlDriverName(),
          s"""
             |schemaspy.db=$dbName
             |schemaspy.host=${propsMap.getOrElse("host", "")}
             |schemaspy.port=${propsMap.getOrElse("port", "")}
           """.stripMargin)
      case _ =>
        throw new SparkException(s"Unknown JDBC driver: $driverName")
    }

    val propFile = generatePropFile {
      val jdbcDriverFile = getJdbcDriverFile(jdbcDriverName)
      (jdbcDriverFile.getAbsolutePath, dbSpecificProps)
    }

    runSchemaSpy(driverTypeName, userDefinedOutputPath, propFile)
  }

  // To call this function, just say a lien below;
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

      val propFile = generatePropFile {
        val jdbcDriverFile = getJdbcDriverFile(getSqliteDriverName())
        val dbFile = dumpSparkCatalog(sparkSession, tables)
        (jdbcDriverFile.getAbsolutePath, s"schemaspy.db=${dbFile.getAbsolutePath}")
      }

      runSchemaSpy("sqlite-xerial", userDefinedOutputPath, propFile, Some(_ => {
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

  // To call this function, just say a lien below;
  // >>> scavenger.constraints().setDbName('default').setTableName('t').infer().show()
  def inferConstraints(userDefinedOutputPath: String, dbName: String, tableName: String): String = {
    withSparkSession { sparkSession =>
      val table = sparkSession.sessionState.catalog.listTables(dbName)
        .find(_.table == tableName).getOrElse {
          throw new SparkException(s"Table '$tableName' does not exist in database '$dbName'.")
        }

      val propFile = generatePropFile {
        val jdbcDriverFile = getJdbcDriverFile(getSqliteDriverName())
        val dbFile = dumpSparkCatalog(sparkSession, table :: Nil)
        (jdbcDriverFile.getAbsolutePath, s"schemaspy.db=${dbFile.getAbsolutePath}")
      }

      runSchemaSpy("sqlite-xerial", userDefinedOutputPath, propFile, Some(_ => {
        generateMetaFile {
          val tck = IntegrityConstraintDiscovery.tableConstraintsKey
          val (tableConstraints, columnConstraints) =
            IntegrityConstraintDiscovery.exec(sparkSession, table.table)
              .map { case (k, v) => k -> v.mkString(",") }
              .partition { v => v._1 == tck }

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
