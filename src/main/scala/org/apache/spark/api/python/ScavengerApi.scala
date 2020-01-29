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

import java.io.{File, FileWriter, PrintWriter}
import java.net.URL
import java.sql.{Connection, Driver, ResultSet, Statement}
import java.util.UUID

import scala.collection.mutable
import org.apache.commons.lang.RandomStringUtils
import org.apache.spark.SparkException
import org.apache.spark.internal.Logging
import org.apache.spark.python.ScavengerConf._
import org.apache.spark.python._
import org.apache.spark.sql._
import org.apache.spark.sql.catalyst.TableIdentifier
import org.apache.spark.sql.catalyst.plans.logical.LeafNode
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._
import org.apache.spark.util.{MutableURLClassLoader, Utils => SparkUtils}

import io.github.maropu.SchemaSpyLauncher
import io.github.maropu.Utils._

/** A Python API entry point to call a SchemaSpy command. */
object ScavengerApi extends Logging {

  private val JDBC_DRIVERS_HOME = "JDBC_DRIVERS_HOME"
  private val JDBC_SQLITE_VERSION = "JDBC_SQLITE_VERSION"
  private val JDBC_POSTGRESQL_VERSION = "JDBC_POSTGRESQL_VERSION"

  private val discreteValueThres = 80
  private val emptyProp = new java.util.Properties()

  private def withSparkSession[T](f: SparkSession => T): T = {
    SparkSession.getActiveSession.map { sparkSession =>
      f(sparkSession)
    }.getOrElse {
      throw new SparkException("An active SparkSession not found.")
    }
  }

  private def withSQLConf[T](pairs: (String, String)*)(f: => T): T= {
    val conf = SQLConf.get
    val (keys, values) = pairs.unzip
    val currentValues = keys.map { key =>
      if (conf.contains(key)) {
        Some(conf.getConfString(key))
      } else {
        None
      }
    }
    (keys, values).zipped.foreach { (k, v) =>
      assert(!SQLConf.staticConfKeys.contains(k))
      conf.setConfString(k, v)
    }
    try f finally {
      keys.zip(currentValues).foreach {
        case (key, Some(value)) => conf.setConfString(key, value)
        case (key, None) => conf.unsetConf(key)
      }
    }
  }

  private def withTempView[T](spark: SparkSession, df: DataFrame, cache: Boolean = false)(f: String => T): T = {
    val tempView = getRandomString("tempView_")
    if (cache) df.cache()
    df.createOrReplaceTempView(tempView)
    val ret = f(tempView)
    spark.sql(s"DROP VIEW $tempView")
    ret
  }

  private def getRandomString(prefix: String = ""): String = {
    s"$prefix${SparkUtils.getFormattedClassName(this)}_${RandomStringUtils.randomNumeric(12)}"
  }

  private def getEnvOrFail(key: String): String = {
    val value = System.getenv(key)
    if (value == null) {
      throw new SparkException(s"'$key' not defined correctly.")
    }
    value
  }

  private def getTempOutputPath(): String = {
    val tempPath = s"${System.getProperty("java.io.tmpdir")}/scavenger-${UUID.randomUUID.toString}.output"
    val tempDir = new File(tempPath)
    if (tempDir.mkdir()) {
      tempDir.getAbsolutePath
    } else {
      throw new SparkException(s"Cannot create a temporary directory: $tempPath")
    }
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

  private def generatePropFile(f: => (String, String)): File = {
    val tempDir = SparkUtils.createTempDir(namePrefix = "scavenger")
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
    val tempDir = SparkUtils.createTempDir(namePrefix = "scavenger")
    val metaFile = new File(tempDir, "meta.xml")
    val metaWriter = new PrintWriter(metaFile, "UTF-8")
    try {
      metaWriter.write(schemaSpyMetaTemplate(tableContent))
    } finally {
      metaWriter.close()
    }
    metaFile
  }

  // To call this function, just say a line below;
  // >>> spySchema('-dbhelp')
  def run(args: String): Unit = {
    SchemaSpyLauncher.run(args.split(" "): _*)
  }

  private def runSchemaSpy(
      jdbcDriverTypeName: String,
      outputPath: String,
      propFile: File,
      doProfileFunc: Option[Unit => File] = None): String = {
    assert(outputPath.nonEmpty, "Output path not specified")
    val additionalConfigs = doProfileFunc.map { f =>
      val metaFile = f()
      Seq("-meta", metaFile.getAbsolutePath)
    }.getOrElse {
      Seq.empty
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

  // To call this function, just say a line below;
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

  // To call this function, just say a line below;
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

    val outputPath = if (userDefinedOutputPath.nonEmpty) {
      userDefinedOutputPath
    } else {
      getTempOutputPath()
    }

    val propFile = generatePropFile {
      val jdbcDriverFile = getJdbcDriverFile(jdbcDriverName)
      (jdbcDriverFile.getAbsolutePath, dbSpecificProps)
    }

    runSchemaSpy(driverTypeName, outputPath, propFile)
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

  // To call this function, just say a line below;
  // >>> scavenger.repair().setDbName('default').setTableName('t').infer()
  def detectErrorCells(constraintFilePath: String, dbName: String, tableName: String, rowIdAttr: String): DataFrame = {
    withSparkSession { sparkSession =>
      val table = sparkSession.sessionState.catalog.listTables(dbName)
        .find(_.table == tableName).getOrElse {
          throw new SparkException(s"Table '$tableName' does not exist in database '$dbName'.")
        }

      // Checks if `table` has a column named `rowIdAttr`
      val tableAttrs = sparkSession.table(table.identifier).schema.map(_.name)
      val tableAttrNum = sparkSession.table(tableName).schema.length
      if (!tableAttrs.contains(rowIdAttr)) {
        throw new SparkException(s"Column '$rowIdAttr' does not exist in table '$tableName'.")
      }

      val (tableRowCnt, discreteAttrs) = {
        val queryToComputeStats = {
          val tableStats = {
            val df = sparkSession.table(tableName)
            val tableNode = df.queryExecution.analyzed.collectLeaves().head.asInstanceOf[LeafNode]
            tableNode.computeStats()
          }
          val attrStatMap = tableStats.attributeStats.map {
            kv => (kv._1.name, kv._2.distinctCount)
          }
          // If we have stats in catalog, we just use them
          val rowCount = tableStats.rowCount.map { cnt => s"bigint(${cnt.toLong}) /* rowCount */" }
            .getOrElse("COUNT(1)")
          val approxCntEnabled = sparkSession.sessionState.conf.fkInferenceApproxCountEnabled
          val distinctCounts = tableAttrs.map { attrName =>
            val aggFunc = if (attrName != rowIdAttr && approxCntEnabled) {
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
             |  $tableName
           """.stripMargin
        }

        logDebug(s"Query to compute $tableName stats:" + queryToComputeStats)

        val statsRow = sparkSession.sql(queryToComputeStats).take(1).head
        val attrsWithStats = tableAttrs.zipWithIndex.map { case (f, i) => (f, statsRow.getLong(i + 1)) }
        val rowCnt = statsRow.getLong(0)
        logWarning(s"rowCnt:$rowCnt ${attrsWithStats.map { case (a, c) => s"distinctCnt($a):$c" }.mkString(" ")}")
        if (attrsWithStats.collectFirst { case (a, cnt) if a == rowIdAttr => cnt }.get != rowCnt) {
          throw new SparkException(s"Uniqueness does not hold in column '$rowIdAttr' of table '$tableName'.")
        }
        def isDiscrete(v: Long) = 1 < v && v < discreteValueThres
        val (discreteCols, nonDiscreteCols) = attrsWithStats.partition(a => isDiscrete(a._2))
        if (nonDiscreteCols.size > 1) {
          logWarning("Dropped the columns having non-suitable domain size: " +
            nonDiscreteCols.filterNot(_._1 == rowIdAttr).map { case (a, c) => s"$a($c)" }.mkString(", "))
        }
        (rowCnt, discreteCols.map(_._1))
      }

      val constraints = {
        // Loads all the denial constraints from a given file path
        val allConstraints = DenialConstraints.parse(constraintFilePath)
        // Checks if all the attributes contained in `constraintFilePath` exist in `table`
        val attrsInConstraints = allConstraints.attrNames
        val discreteAttrSet = discreteAttrs.toSet
        val absentAttrs = attrsInConstraints.filterNot(discreteAttrSet.contains)
        if (absentAttrs.nonEmpty) {
          logWarning(s"Non-existent constraint attributes found in $tableName: " +
            absentAttrs.mkString(", "))
          val newPredEntries = allConstraints.entries.filter { _.forall { p =>
            discreteAttrSet.contains(p.leftAttr) && discreteAttrSet.contains(p.rightAttr)
          }}
          if (newPredEntries.isEmpty) {
            throw new SparkException(s"No valid constraint found in $tableName")
          }
          allConstraints.copy(entries = newPredEntries)
        } else {
          allConstraints
        }
      }

      // Detects error erroneous cells in a given table
      val tableAttrToId = discreteAttrs.zipWithIndex.toMap
      val dfs = constraints.entries.flatMap { preds =>
        val queryToValidateConstraint =
          s"""
             |SELECT t1.$rowIdAttr `_tid_`
             |FROM $tableName AS t1
             |WHERE EXISTS (
             |  SELECT t2.$rowIdAttr
             |  FROM $tableName AS t2
             |  WHERE ${DenialConstraints.toWhereCondition(preds, "t1", "t2")}
             |)
           """.stripMargin

        val df = sparkSession.sql(queryToValidateConstraint)
        logDebug(
          s"""
             |Number of violate tuples: ${df.count}
             |Query to validate constraints:
             |$queryToValidateConstraint
           """.stripMargin)

        val attrs = preds.flatMap { p => p.leftAttr :: p.rightAttr :: Nil }.map { attr =>
          val attrName = functions.lit(attr)
          val attrId = tableAttrToId(attr)
          df.withColumn("attrName", attrName).selectExpr("_tid_", "attrName", s"int(_tid_) * int($tableAttrNum) + int($attrId) as cellId", s"int($attrId) attr_idx")
        }
        attrs
      }

      withTempView(sparkSession, dfs.reduce(_.union(_)).distinct(), cache = true) { errCellView =>
        val dkCellsView = "DkCellsView"
        sparkSession.table(errCellView).createOrReplaceTempView(dkCellsView)
        logWarning(s"Exposed dirty cells as $dkCellsView")

        val attrsToRepair = {
          val tgtAttrs = sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
            .collect.head.getSeq[String](0)
          val discreteTgtAttrs = tgtAttrs.filter(discreteAttrs.contains)
          if (discreteTgtAttrs.size != tgtAttrs.size) {
            logWarning(s"Dropped the attributes having too large domains: " +
              tgtAttrs.filterNot(discreteAttrs.contains).mkString(", "))
          }
          if (discreteTgtAttrs.isEmpty) {
            throw new SparkException(s"No target attribute to repair in $table.")
          }
          discreteTgtAttrs
        }

        logWarning(s"Start repairing ${attrsToRepair.size} attributes " +
          s"(${attrsToRepair.mkString(", ")}) in " +
          s"$tableName(${discreteAttrs.mkString(", ")})")

        // Computes numbers for single and pair-wise statistics in the input table
        val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
          discreteAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
        }

        val statsDf = {
          val queryToComputeStats =
            s"""
               |SELECT ${discreteAttrs.mkString(", ")}, COUNT(1) cnt
               |FROM $table
               |GROUP BY GROUPING SETS (
               |  ${discreteAttrs.map(a => s"($a)").mkString(", ")},
               |  ${attrPairsToRepair.map { case (a1, a2) => s"($a1,$a2)" }.mkString(", ")}
               |)
             """.stripMargin
          logDebug(queryToComputeStats)
          sparkSession.sql(queryToComputeStats)
        }

        def whereCaluseToFilter(a: String): String =
          s"$a IS NOT NULL AND ${discreteAttrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"

        val corrAttrs = withTempView(sparkSession, statsDf, cache = true) { statsView =>
          val pairWiseStats = attrPairsToRepair.map { case attrPair @ (attrToRepair, a) =>
            val pairWiseStatDf = sparkSession.sql(
              s"""
                 |SELECT
                 |  v1.X, v1.Y, (cntX / $tableRowCnt) pX, (cntY / $tableRowCnt) pY, (cntXY / $tableRowCnt) pXY
                 |FROM (
                 |  SELECT $attrToRepair X, $a Y, cnt cntXY
                 |  FROM $statsView
                 |  WHERE $attrToRepair IS NOT NULL AND
                 |    $a IS NOT NULL
                 |) v1, (
                 |  SELECT $attrToRepair X, cnt cntX
                 |  FROM $statsView
                 |  WHERE ${whereCaluseToFilter(attrToRepair)}
                 |) v2, (
                 |  /* Use `MAX` to drop ($a, null) tuples in `$tableName` */
                 |  SELECT $a Y, MAX(cnt) cntY
                 |  FROM $statsView
                 |  WHERE ${whereCaluseToFilter(a)}
                 |  GROUP BY $a
                 |) v3
                 |WHERE
                 |  v1.X = v2.X AND
                 |  v1.Y = v3.Y
               """.stripMargin)

            // val pairWiseStatsView = s"PairWiseStatView_${attrToRepair}_$a"
            // pairWiseStatDf.createOrReplaceTempView(pairWiseStatsView)
            // logWarning(s"Exposed pair-wise statistics as $pairWiseStatsView")

            attrPair -> pairWiseStatDf.selectExpr("SUM(pXY * log2(pXY / (pX * pY)))")
              .collect.map { row =>
                if (!row.isNullAt(0)) row.getDouble(0) else 0.0
              }.head
          }
          pairWiseStats.groupBy { case ((attrToRepair, _), _) => attrToRepair }.map { case (k, v) =>
            k -> v.map { case ((_, attr), v) => (attr, v) }.sortBy(_._2).reverse.take(2)
          }
        }

        val statsView = "StatsView"
        statsDf.createOrReplaceTempView(statsView)
        logWarning(s"Exposed statistics as $statsView")

        sparkSession.udf.register("extractField", (row: Row, offset: Int) => row.getString(offset))
        val cellExprs = discreteAttrs.map { a => s"CAST(l.$a AS STRING) $a" }
        val rvDf = sparkSession.sql(
          s"""
             |SELECT
             |  l.$rowIdAttr,
             |  ${cellExprs.mkString(", ")},
             |  cellId,
             |  attr_idx,
             |  attrName,
             |  extractField(struct(${cellExprs.mkString(", ")}), attr_idx) initValue
             |FROM
             |  $tableName l, $errCellView r
             |WHERE
             |  l.$rowIdAttr = r._tid_
           """.stripMargin)

        val rvRdd = rvDf.rdd.zipWithIndex().map { case (r, i) => Row.fromSeq(i +: r.toSeq) }
        val rvSchemaWithId = StructType(StructField("_eid", LongType) +: rvDf.schema)
        val rvDf_ = sparkSession.createDataFrame(rvRdd, rvSchemaWithId)

        val rvView = "RvView"
        rvDf_.createOrReplaceTempView(rvView)
        logWarning(s"Exposed random variables as $rvView")

        val rvDfs = withTempView(sparkSession, rvDf_, cache = true) { rvView =>
          withSQLConf(SQLConf.CROSS_JOINS_ENABLED.key -> "true") {
            corrAttrs.map { case (attrName, Seq((a1, _), (a2, _))) =>
              sparkSession.sql(
                s"""
                   |SELECT
                   |  _eid vid,
                   |  $rowIdAttr,
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
                   |    rv.$rowIdAttr,
                   |    rv.cellId,
                   |    rv.attr_idx,
                   |    rv.attrName,
                   |    array_sort(array_union(array(rv.initValue), d.domain)) domain,
                   |    IF(ISNULL(rv.initValue), shuffle(d.domain)[0], rv.initValue) initValue
                   |  FROM
                   |    $rvView rv, (
                   |    SELECT $a1, $a2, concat(dom1, dom2) domain
                   |    FROM (
                   |      SELECT $a1, collect_set($attrName) dom1
                   |      FROM $table
                   |      GROUP BY $a1
                   |    ), (
                   |      SELECT $a2, collect_set($attrName) dom2
                   |      FROM $table
                   |      GROUP BY $a2
                   |    )
                   |  ) d
                   |  WHERE
                   |    rv.attrName = "$attrName" AND
                   |    rv.$a1 = d.$a1 AND
                   |    rv.$a2 = d.$a2
                   |)
                 """.stripMargin)
            }
          }
        }

        val constraintFtDfs = withTempView(sparkSession, rvDfs.reduce(_.union(_)), cache = true) { cellDomain =>
          val cellDomainView = "CellDomainView"
          sparkSession.table(cellDomain).createOrReplaceTempView(cellDomainView)
          logWarning(s"Exposed cell domain as $cellDomainView")

          val weakLabelDf = sparkSession.sql(
            s"""
               |SELECT vid, weakLabel, weakLabelIndex, fixed, /* (t2.cellId IS NULL) */ IF(rand() > 0.7, true, false) AS clean
               |FROM $cellDomainView AS t1
               |LEFT JOIN $dkCellsView AS t2
               |ON t1.cellId = t2.cellId
               |WHERE weakLabel IS NOT NULL AND (
               |  t2.cellId IS NULL OR t1.fixed != 1
               |)
             """.stripMargin)
          val weakLabelView = "WeakLabelView"
          weakLabelDf.createOrReplaceTempView(weakLabelView)
          logWarning(s"Exposed weak labels as $weakLabelView")

          val varMaskDf = sparkSession.sql(s"SELECT vid, domainSize FROM $cellDomainView")
          val varMaskView = "VarMaskView"
          varMaskDf.createOrReplaceTempView(varMaskView)
          logWarning(s"Exposed variable masks as $varMaskView")

          val (totalVars, classes) = sparkSession.sql(s"SELECT COUNT(vid), MAX(domainSize) FROM $cellDomain")
            .collect.headOption.map { case Row(l: Long, i: Int) => (l, i) }.get
          logWarning(s"totalVars=$totalVars classes=$classes attrNum=$tableAttrNum")

          // PyTorch feature:
          // tensor = -1 * torch.ones(1, classes, attrNum)
          // tensor[0][init_idx][attr_idx] = 1.0
          val initAttrFtDf = sparkSession.sql(
               s"SELECT initIndex init_idx, attr_idx FROM $cellDomain")
          val initAttrFeatures = "InitAttrFeatureView"
          initAttrFtDf.createOrReplaceTempView(initAttrFeatures)
          logWarning(s"Exposed features as $initAttrFeatures")

          // PyTorch feature: torch.zeros(1, classes, attrName) = prob
          val freqFtDf = attrsToRepair.map { attr =>
            sparkSession.sql(
              s"""
                 |SELECT vid, valId idx, attr_idx, (freq / $tableRowCnt) prob
                 |FROM (
                 |  SELECT vid, attr_idx, posexplode(domain) (valId, rVal)
                 |  FROM $cellDomain
                 |) d, (
                 |  SELECT $attr, COUNT(1) freq
                 |  FROM $table
                 |  GROUP BY $attr
                 |) f
                 |WHERE
                 |  d.rVal = f.$attr
               """.stripMargin)
          }.reduce(_.union(_))
          val freqFeatures = "FreqFeatureView"
          freqFtDf.createOrReplaceTempView(freqFeatures)
          logWarning(s"Exposed features as $freqFeatures")

          val occFtDf = withTempView(sparkSession, statsDf) { statsView =>
            attrsToRepair.indices.flatMap { i =>
              val (Seq((rvAttr, _)), attrs) = attrsToRepair.zipWithIndex.partition { case (_, j) => i == j }
              attrs.map { case (attr, _) =>
                val index = tableAttrToId(rvAttr) * tableAttrNum + tableAttrToId(attr)
                // PyTorch feature: torch.zeros(1, classes, attrName * attrName) = prob
                sparkSession.sql(
                  s"""
                     |SELECT
                     |  vid, valId rv_domain_idx, $index index, (cntYX / $tableRowCnt) pYX, (cntX / $tableRowCnt) pX, prob
                     |FROM (
                     |  SELECT vid, valId, rVal, $attr
                     |  FROM
                     |    $table t, (
                     |      SELECT vid, $rowIdAttr, posexplode(domain) (valId, rVal)
                     |      FROM $cellDomain
                     |      WHERE attrName = '$rvAttr'
                     |    ) d
                     |  WHERE
                     |    t.$rowIdAttr = d.$rowIdAttr
                     |) t1, (
                     |  SELECT YX.$rvAttr, X.$attr, cntYX, cntX, (cntYX / cntX) prob
                     |  FROM (
                     |    SELECT $rvAttr, $attr X, cnt cntYX
                     |    FROM $statsView
                     |    WHERE $rvAttr IS NOT NULL AND
                     |      $attr IS NOT NULL
                     |  ) YX, (
                     |    SELECT $attr, cnt cntX
                     |    FROM $statsView
                     |    WHERE ${whereCaluseToFilter(attr)}
                     |  ) X
                     |  WHERE YX.X = X.$attr
                     |) t2
                     |WHERE
                     |  t1.rVal = t2.$rvAttr AND
                     |  t1.$attr = t2.$attr
                   """.stripMargin)
              }
            }.reduce(_.union(_))
          }
          val occurAttrFeatures = "OccurAttrFeatureView"
          occFtDf.orderBy("vid").createOrReplaceTempView(occurAttrFeatures)
          logWarning(s"Exposed features as $occurAttrFeatures")

          val posValDf = sparkSession.sql(
            s"""
               |SELECT vid, $rowIdAttr, cellId, attrName, posexplode(domain) (valId, rVal)
               |FROM $cellDomain
             """.stripMargin)
          withTempView(sparkSession, posValDf) { posValues =>
            val posValuesView = "PosValuesView"
            sparkSession.table(posValues).createOrReplaceTempView(posValuesView)
            logWarning(s"Exposed pos values as $posValuesView")

            val offsets = constraints.entries.scanLeft(0) { case (idx, preds) => idx + preds.size }.init
            val queries = constraints.entries.zip(offsets).flatMap { case (preds, offset) =>
              preds.indices.map { i =>
                val (Seq((violationPred, _)), fixedPreds) = preds.zipWithIndex.partition { case (_, j) => i == j }
                val fixedWhereCaluses = DenialConstraints.toWhereCondition(fixedPreds.map(_._1), "t1", "t2")
                val rvAttr = violationPred.rightAttr
                // PyTorch feature: torch.zeros(totalVars,classes,1) = #violations
                val queryToCountViolations =
                  s"""
                     |SELECT
                     |  ${offset + i} constraintId, vid, valId, COUNT(1) violations
                     |FROM
                     |  $table as t1, $table as t2, $posValues as t3
                     |WHERE
                     |  t1.$rowIdAttr != t2.$rowIdAttr AND
                     |  t1.$rowIdAttr = t3.$rowIdAttr AND
                     |  t3.attrName = '$rvAttr' AND
                     |  $fixedWhereCaluses AND
                     |  t3.rVal = t2.$rvAttr
                     |GROUP BY vid, valId
                   """.stripMargin

                logDebug(queryToCountViolations)
                queryToCountViolations
              }
            }

            queries.zipWithIndex.map { case (q, i) =>
              outputConsole(s"Starts processing the $i/${queries.size} query to compute #violations...")
              sparkSession.sql(q)
            }
          }
        }
        val constraintFeatures = "ConstraintFeatureView"
        constraintFtDfs.reduce(_.union(_)).createOrReplaceTempView(constraintFeatures)
        logWarning(s"Exposed features as $constraintFeatures")

        sparkSession.emptyDataFrame
      }
    }
  }
}
