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

  private val maxAttrNumToRepair = 32
  private val discreteValueThres = 80
  private val minMaxAttrNumToComputeDomain = (2, 4)
  private val minCorrValueToComputeDomain = 10.0
  private val sampleRatioToComputeStats = 0.30
  private val sampleRatioToCountViolations = 0.30
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

  private case class Metadata(spark: SparkSession) {
    private val kvs = mutable.ArrayBuffer[(String, Any)]()

    def add(key: String, value: Any): Unit = {
      kvs += key -> value
    }

    override def toString: String = {
      kvs.map {
        case (k, v: String) => s"""$k=>"$v""""
        case (k, ar: Seq[String]) => s"$k=>${ar.map(v => s""""$v"""").mkString(",")}"
      }.mkString(", ")
    }

    def toJson: String = {
      kvs.map {
        case (k, v: String) => s""""$k":"$v""""
        case (k, ar: Seq[String]) => s""""$k":${ar.map(v => s""""$v"""").mkString("[", ",", "]")}"""
      }.mkString("{", ",", "}")
    }
  }

  private def timer[R](name: String)(block: => R): R = {
    val t0 = System.nanoTime()
    val result = block
    val t1 = System.nanoTime()
    logWarning(s"Elapsed time to compute '$name': " + ((t1 - t0 + 0.0) / 1000000000.0)+ "s")
    result
  }

  private def createTempView(df: DataFrame): String = {
    val tempViewId = getRandomString()
    val numShufflePartitions = df.sparkSession.sessionState.conf.numShufflePartitions
    df.coalesce(numShufflePartitions).cache.createOrReplaceTempView(tempViewId)
    timer(tempViewId) {
      // TODO: This dummy code to invoke a job can be replaced in v3.0 with
      // `df.write.format("noop").mode("overwrite").save()`
      df.sparkSession.table(tempViewId).groupBy().count().foreach(_ => {})
    }
    tempViewId
  }

  def prepareInputTable(dbName: String, tableName: String, rowId: String): String = {
     withSparkSession { sparkSession =>
      // Checks if the given table has a column named `rowId`
      val inputDf = sparkSession.table(if (dbName.nonEmpty) s"$dbName.$tableName" else tableName)
      val tableAttrs = inputDf.schema.map(_.name)
      if (!tableAttrs.contains(rowId)) {
        // TODO: Implicitly adds unique row IDs if they don't exist in a given table
        throw new SparkException(s"Column '$rowId' does not exist in table '$dbName.$tableName'.")
      }

      val discreteAttrs = {
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
          val approxCntEnabled = sparkSession.sessionState.conf.fkInferenceApproxCountEnabled
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

        logDebug(s"Query to compute $dbName.$tableName stats:" + queryToComputeStats)

        val statsRow = sparkSession.sql(queryToComputeStats).take(1).head
        val attrsWithStats = tableAttrs.zipWithIndex.map { case (f, i) => (f, statsRow.getLong(i + 1)) }
        val rowCnt = statsRow.getLong(0)

        logDebug({
          val distinctCnts = attrsWithStats.map { case (a, c) => s"distinctCnt($a):$c" }
          s"rowCnt:$rowCnt ${distinctCnts.mkString(" ")}"
        })

        if (attrsWithStats.collectFirst { case (a, cnt) if a == rowId => cnt }.get != rowCnt) {
          throw new SparkException(s"Uniqueness does not hold in column '$rowId' of table '$dbName.$tableName'.")
        }
        def isDiscrete(v: Long) = 1 < v && v < discreteValueThres
        val (discreteCols, nonDiscreteCols) = attrsWithStats.partition(a => isDiscrete(a._2))
        if (nonDiscreteCols.size > 1) {
          logWarning("Dropped the columns having non-suitable domain size: " +
            nonDiscreteCols.filterNot(_._1 == rowId).map { case (a, c) => s"$a($c)" }.mkString(", "))
        }
        if (discreteCols.size >= maxAttrNumToRepair) {
          throw new SparkException(s"Maximum number of attributes is $maxAttrNumToRepair, but " +
            s"the ${discreteCols.size} discrete attributes found in table '$dbName.$tableName'")
        }
        discreteCols.map(_._1)
      }

      val df = inputDf.selectExpr(discreteAttrs :+ rowId: _*)
      createTempView(df)
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
      logWarning(s"Non-existent constraint attributes found in $tableName: " +
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
    withSparkSession { sparkSession =>
      // Checks if the given table has a column named `rowId`
      val df = sparkSession.table(if (dbName.nonEmpty) s"$dbName.$tableName" else tableName)
      val tableAttrs = df.schema.map(_.name)
      if (!tableAttrs.contains(rowId)) {
        // TODO: Implicitly adds unique row IDs if they don't exist in a given table
        throw new SparkException(s"Column '$rowId' does not exist in table '$dbName.$tableName'.")
      }

      withTempView(sparkSession, df, cache = true) { inputTableView =>
        val tableAttrs = sparkSession.table(inputTableView).schema.map(_.name)
        val tableAttrNum = sparkSession.table(inputTableView).schema.length
        val constraints = loadConstraintsFromFile(constraintFilePath, tableName, tableAttrs)

        // Detects error erroneous cells in a given table
        val tableAttrToId = tableAttrs.zipWithIndex.toMap
        val errCellDf = constraints.entries.flatMap { preds =>
          val queryToValidateConstraint =
            s"""
               |SELECT t1.$rowId `_tid_`
               |FROM $inputTableView AS t1
               |WHERE EXISTS (
               |  SELECT t2.$rowId
               |  FROM $inputTableView AS t2
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

          preds.flatMap { p => p.leftAttr :: p.rightAttr :: Nil }.map { attr =>
            val attrId = tableAttrToId(attr)
            df.selectExpr("_tid_",
              s""""$attr" AS attrName""",
              s"int(_tid_) * int($tableAttrNum) + int($attrId) AS cellId",
              s"int($attrId) AS attr_idx")
          }
        }.reduce(_.union(_)).distinct()

        val errCellView = createTempView(errCellDf)
        outputConsole({
          val tableRowCnt = sparkSession.table(inputTableView).count()
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

  def computeAttrStats(inputTableView: String, errCellView: String, rowId: String): String = {
    withSparkSession { sparkSession =>
      // Computes numbers for single and pair-wise statistics in the input table
      val tableAttrs = sparkSession.table(inputTableView).schema.map(_.name)
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      val attrPairsToRepair = attrsToRepair.flatMap { attrToRepair =>
        tableAttrs.filter(attrToRepair != _).map(a => (attrToRepair, a))
      }

      val statsDf = {
        val groupSets = attrPairsToRepair.map(p => Set(p._1, p._2)).distinct
        val sampleInputDf = sparkSession.table(inputTableView).sample(sampleRatioToComputeStats)
        withTempView(sparkSession, sampleInputDf) { sampleInputView =>
          sparkSession.sql(
            s"""
               |SELECT ${tableAttrs.mkString(", ")}, COUNT(1) cnt
               |FROM $sampleInputView
               |GROUP BY GROUPING SETS (
               |  ${tableAttrs.map(a => s"($a)").mkString(", ")},
               |  ${groupSets.map(_.toSeq).map { case Seq(a1, a2) => s"($a1,$a2)" }.mkString(", ")}
               |)
             """.stripMargin)
        }
      }

      createTempView(statsDf)
    }
  }

  def computeMetadata(inputTableView: String, statsView: String, errCellView: String, rowId: String): String = {
    withSparkSession { sparkSession =>
      val metadata = Metadata(sparkSession)
      val discreteAttrs = sparkSession.table(inputTableView).schema.map(_.name).filter(_ != rowId)
      val tableRowCnt = sparkSession.table(inputTableView).count()

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
             |  FROM $statsView
             |  WHERE $attrToRepair IS NOT NULL AND
             |    $a IS NOT NULL
             |) v1, (
             |  SELECT $attrToRepair X, cnt cntX
             |  FROM $statsView
             |  WHERE ${whereCaluseToFilterStat(attrToRepair)}
             |) v2, (
             |  /* TODO: Needs to reconsider how-to-handle NULL */
             |  /* Use `MAX` to drop ($a, null) tuples in `$inputTableView` */
             |  SELECT $a Y, MAX(cnt) cntY
             |  FROM $statsView
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
      logWarning({
        val pairStats = pairWiseStatMap.map { case (k, v) =>
          s"$k(${v.head._2},${v.last._2})=>${v.map(a => s"${a._1}:${a._2}").mkString(",")}"
        }
        s"""
           |Pair-wise statistics:
           |${pairStats.mkString("\n")}
         """.stripMargin
      })

      sparkSession.udf.register("extractField", (row: Row, offset: Int) => row.getString(offset))
      val cellExprs = discreteAttrs.map { a => s"CAST(l.$a AS STRING) $a" }
      val rvDf = sparkSession.sql(
        s"""
           |SELECT
           |  l.$rowId,
           |  ${cellExprs.mkString(", ")},
           |  cellId,
           |  attr_idx,
           |  attrName,
           |  extractField(struct(${cellExprs.mkString(", ")}), attr_idx) initValue
           |FROM
           |  $inputTableView l, $errCellView r
           |WHERE
           |  l.$rowId = r._tid_
         """.stripMargin)

      // TODO: More efficient way to assign unique IDs
      val rvWithIdDf = {
        val rvRdd = rvDf.rdd.zipWithIndex().map { case (r, i) => Row.fromSeq(i +: r.toSeq) }
        val rvSchemaWithId = StructType(StructField("_eid", LongType) +: rvDf.schema)
        sparkSession.createDataFrame(rvRdd, rvSchemaWithId)
      }

      // TODO: Currently, pick up two co-related attributes only
      val corrAttrs = pairWiseStatMap.map { case (k, v) =>
        val (minAttrNum, maxAttrNum) = minMaxAttrNumToComputeDomain
        val attrs = v.filter(_._2 > minCorrValueToComputeDomain)
        (k, if (attrs.size > maxAttrNum) {
          attrs.take(maxAttrNum)
        } else if (attrs.size < minAttrNum) {
          // TODO: If correlated attributes not found, we need to pick up data
          // from its domain randomly.
          logWarning(s"Correlated attributes not found for $k")
          v.take(minAttrNum)
        } else {
          attrs
        })
      }
      val cellDomainDf = withTempView(sparkSession, rvWithIdDf) { rvView =>
        corrAttrs.map { case (attrName, corrAttrsWithScores) =>
          assert(corrAttrsWithScores.size >= 2)
          // Computes domains for error cells
          val corrAttrs = corrAttrsWithScores.map(_._1)
          logWarning(s"Computing '$attrName' domain from correlated attributes (${corrAttrs.mkString(",")})...")
          val dfs = corrAttrs.zipWithIndex.map { case (attr, i) =>
            sparkSession.sql(
              s"""
                 |SELECT $attr, collect_set($attrName) dom$i
                 |FROM $inputTableView
                 |GROUP BY $attr
               """.stripMargin)
          }
          val domainSpaceDf = dfs.tail.foldLeft(dfs.head) { case (a, b) => a.join(b) }
          withTempView(sparkSession, domainSpaceDf) { domainSpaceView =>
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
                 |  FROM
                 |    $rvView rv, (
                 |      SELECT ${corrAttrs.mkString(", ")}, concat(${corrAttrs.indices.map(i => s"dom$i").mkString(", ")}) domain
                 |      FROM $domainSpaceView
                 |    ) d
                 |  WHERE
                 |    rv.attrName = "$attrName" AND
                 |    ${corrAttrs.map(v => s"rv.$v = d.$v").mkString(" AND ")}
                 |)
               """.stripMargin)
          }
        }
      }.reduce(_.union(_))

      val cellDomainView = createTempView(cellDomainDf)
      metadata.add("__cell_domain", cellDomainView)

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

      metadata.add("__weak_label", createTempView(weakLabelDf))

      val varMaskDf = sparkSession.sql(s"SELECT vid, domainSize FROM $cellDomainView")
      metadata.add("__var_mask", createTempView(varMaskDf))

      val (totalVars, classes) = sparkSession.sql(s"SELECT COUNT(vid), MAX(domainSize) FROM $cellDomainView")
        .collect.headOption.map { case Row(l: Long, i: Int) => (l, i) }.get

      logWarning(s"totalVars=$totalVars classes=$classes " +
        s"featureAttrs(${discreteAttrs.size})=${discreteAttrs.mkString(",")}")
      metadata.add("totalVars", s"$totalVars")
      metadata.add("classes", s"$classes")
      metadata.add("featureAttrNum", s"${discreteAttrs.size}")
      metadata.add("featureAttrs", discreteAttrs)

      val posValDf = sparkSession.sql(
        s"""
           |SELECT vid, $rowId, cellId, attrName, posexplode(domain) (valId, rVal)
           |FROM $cellDomainView
         """.stripMargin)

      metadata.add("__pos_value", createTempView(posValDf))
      metadata.toJson
    }
  }

  def createInitAttrFeatureView(cellDomainView: String): String = {
    withSparkSession { sparkSession =>
      // PyTorch feature:
      // tensor = -1 * torch.ones(1, classes, attrNum)
      // tensor[0][init_idx][attr_idx] = 1.0
      val initAttrFtDf = sparkSession.sql(s"SELECT initIndex init_idx, attr_idx FROM $cellDomainView")
      createTempView(initAttrFtDf)
    }
  }

  def createFreqFeatureView(inputTableView: String, cellDomainView: String, errCellView: String): String = {
    withSparkSession { sparkSession =>
      val tableRowCnt = sparkSession.table(inputTableView).count()
      val attrsToRepair = {
        sparkSession.sql(s"SELECT collect_set(attrName) FROM $errCellView")
          .collect.head.getSeq[String](0)
      }
      // PyTorch feature: torch.zeros(1, classes, attrName) = prob
      val freqFtDf = attrsToRepair.map { attr =>
        sparkSession.sql(
          s"""
             |SELECT vid, valId idx, attr_idx, (freq / $tableRowCnt) prob
             |FROM (
             |  SELECT vid, attr_idx, posexplode(domain) (valId, rVal)
             |  FROM $cellDomainView
             |) d, (
             |  SELECT $attr, COUNT(1) freq
             |  FROM $inputTableView
             |  GROUP BY $attr
             |) f
             |WHERE
             |  d.rVal = f.$attr
           """.stripMargin)
      }.reduce(_.union(_))

      createTempView(freqFtDf)
    }
  }

  private def whereCaluseToFilterStat(a: String, attrs: Seq[String]): String = {
    s"$a IS NOT NULL AND ${attrs.filter(_ != a).map(a => s"$a IS NULL").mkString(" AND ")}"
  }

  def createOccurAttrFeatureView(inputTableView: String, errCellView: String, cellDomainView: String, statsView: String, rowId: String): String = {
    withSparkSession { sparkSession =>
      val tableAttrs = sparkSession.table(inputTableView).schema.map(_.name)
      val tableAttrNum = sparkSession.table(inputTableView).schema.length
      val tableRowCnt = sparkSession.table(inputTableView).count()
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
          // PyTorch feature: torch.zeros(1, classes, attrName * attrName) = prob
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
               |    $inputTableView t, (
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
               |    FROM $statsView
               |    WHERE $rvAttr IS NOT NULL AND
               |      $attr IS NOT NULL
               |  ) YX, (
               |    /* Use `MAX` to drop ($attr, null) tuples in `$inputTableView` */
               |    SELECT $attr, MAX(cnt) cntX
               |    FROM $statsView
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

      createTempView(occFtDf)
    }
  }

  def createConstraintFeatureView(constraintFilePath: String, inputTableView: String, posValuesView: String, rowId: String): String = {
    withSparkSession { sparkSession =>
      val sampleInputDf = sparkSession.table(inputTableView).sample(sampleRatioToCountViolations)

      withTempView(sparkSession, sampleInputDf, cache = true) { sampleInputView =>
        val metadata = Metadata(sparkSession)
        val predicates = mutable.ArrayBuffer[(String, String)]()
        val tableAttrs = sparkSession.table(inputTableView).schema.map(_.name)
        val constraints = loadConstraintsFromFile(constraintFilePath, inputTableView, tableAttrs)
        val offsets = constraints.entries.scanLeft(0) { case (idx, preds) => idx + preds.size }.init
        val queries = constraints.entries.zip(offsets).flatMap { case (preds, offset) =>
          preds.indices.map { i =>
            val (Seq((violationPred, _)), fixedPreds) = preds.zipWithIndex.partition { case (_, j) => i == j }
            val fixedWhereCaluses = DenialConstraints.toWhereCondition(fixedPreds.map(_._1), "t1", "t2")
            predicates += ((fixedWhereCaluses, violationPred.toString("t1", "t2")))
            val rvAttr = violationPred.rightAttr
            // PyTorch feature: torch.zeros(totalVars,classes,1) = #violations
            val queryToCountViolations =
              s"""
                 |SELECT
                 |  ${offset + i} constraintId, vid, valId, COUNT(1) violations
                 |FROM
                 |  $sampleInputView as t1, $sampleInputView as t2, $posValuesView as t3
                 |WHERE
                 |  t1.$rowId != t2.$rowId AND
                 |  t1.$rowId = t3.$rowId AND
                 |  t3.attrName = '$rvAttr' AND
                 |  $fixedWhereCaluses AND
                 |  t3.rVal = t2.$rvAttr
                 |GROUP BY vid, valId
               """.stripMargin

            logDebug(queryToCountViolations)
            queryToCountViolations
          }
        }
        val constraintFtDf = queries.zipWithIndex.map { case (q, i) =>
          outputConsole(s"Starts processing the $i/${queries.size} query to compute #violations...")
          sparkSession.sql(q)
        }.reduce(_.union(_))

        metadata.add("__constraint_feature", createTempView(constraintFtDf))
        metadata.add("__fixed_preds", predicates.map(_._1))
        metadata.add("__violation_preds", predicates.map(_._2))

        metadata.toJson
      }
    }
  }
}
